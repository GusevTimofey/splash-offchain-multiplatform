use crate::node::NodeConfig;
use crate::tx_tracker::TxTracker;
use async_stream::stream;
use cardano_submit_api::client::{Error, LocalTxSubmissionClient};
use cml_core::serialization::Serialize;
use futures::channel::{mpsc, oneshot};
use futures::{SinkExt, Stream, StreamExt};
use log::{trace, warn};
use pallas_network::miniprotocols::localstate::queries_v16::TransactionInput;
use pallas_network::miniprotocols::localtxsubmission::{
    ApplyTxError, ConwayLedgerFailure, ConwayUtxoWPredFailure, Response, TxValidationError, UtxoFailure,
};
use spectrum_cardano_lib::OutputRef;
use spectrum_offchain::network::Network;
use spectrum_offchain::tx_hash::CanonicalHash;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};

pub struct TxSubmissionAgent<const ERA: u16, Tx, Tracker> {
    client: LocalTxSubmissionClient<ERA, Tx>,
    mailbox: mpsc::Receiver<SubmitTx<Tx>>,
    tracker: Tracker,
    node_config: NodeConfig,
}

impl<const ERA: u16, Tx, Tracker> TxSubmissionAgent<ERA, Tx, Tracker> {
    pub async fn new(
        tracker: Tracker,
        node_config: NodeConfig,
        buffer_size: usize,
    ) -> Result<(Self, TxSubmissionChannel<ERA, Tx>), Error> {
        let tx_submission_client =
            LocalTxSubmissionClient::init(node_config.path.clone(), node_config.magic).await?;
        let (snd, recv) = mpsc::channel(buffer_size);
        let agent = Self {
            client: tx_submission_client,
            mailbox: recv,
            tracker,
            node_config,
        };
        Ok((agent, TxSubmissionChannel(snd)))
    }

    pub async fn restarted(self) -> Result<Self, Error> {
        trace!("Restarting TxSubmissionProtocol");
        let TxSubmissionAgent {
            client,
            mailbox,
            tracker,
            node_config,
        } = self;
        client.close().await;
        let new_tx_submission_client =
            LocalTxSubmissionClient::init(node_config.path.clone(), node_config.magic).await?;
        Ok(Self {
            client: new_tx_submission_client,
            mailbox,
            tracker,
            node_config,
        })
    }
}

pub struct TxSubmissionChannel<const ERA: u16, Tx>(mpsc::Sender<SubmitTx<Tx>>);

impl<const ERA: u16, Tx> Clone for TxSubmissionChannel<ERA, Tx> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Called by the agent right before it writes a tx to the node connection,
/// after it took the request from its mailbox. `true`: the agent writes the
/// tx. `false`: the agent does not write it and drops the responder, so the
/// caller sees [`AgentGone`]. The hook runs before the first byte of the
/// tx-submission message goes to the socket, so a caller can record that the
/// write started.
pub type WriteHook = Box<dyn FnOnce() -> bool + Send + 'static>;

pub struct SubmitTx<Tx>(Tx, oneshot::Sender<SubmissionResult>, Option<WriteHook>);

/// The agent did not answer a request: it ended (its task stopped or
/// panicked) or its write hook refused the write. Whether the tx was written
/// is known only to the hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentGone;

impl Display for AgentGone {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("the tx-submission agent ended before it answered")
    }
}

impl<const ERA: u16, Tx> TxSubmissionChannel<ERA, Tx> {
    /// Submit `tx` with a write hook. The agent calls `hook` right before it
    /// writes the tx, and writes only when the hook returns `true`. Never
    /// panics: a closed mailbox or a dropped responder is [`AgentGone`].
    pub async fn submit_tx_hooked(&mut self, tx: Tx, hook: WriteHook) -> Result<SubmissionResult, AgentGone> {
        let (snd, recv) = oneshot::channel();
        self.0
            .send(SubmitTx(tx, snd, Some(hook)))
            .await
            .map_err(|_| AgentGone)?;
        recv.await.map_err(|_| AgentGone)
    }
}

#[derive(Debug, Clone)]
pub enum SubmissionResult {
    Ok,
    TxRejected { errors: TxRejection },
}

impl From<SubmissionResult> for Result<(), TxRejection> {
    fn from(value: SubmissionResult) -> Self {
        match value {
            SubmissionResult::Ok => Ok(()),
            SubmissionResult::TxRejected { errors } => Err(errors.into()),
        }
    }
}

/// The one call the agent loop makes on its node connection. The production
/// client is [`LocalTxSubmissionClient`]; the tests of this module use a
/// scripted one.
trait SubmitClient<Tx> {
    async fn submit(&mut self, tx: Tx) -> Result<Response<TxValidationError>, Error>;
}

/// The node connection of a running agent. When the agent stream is dropped
/// (its task ended or was aborted), the plexer tasks are aborted too, so the
/// socket closes. A dropped `RunningPlexer` only detaches its tasks, and the
/// muxer task keeps the write half open.
struct ClosingClient<const ERA: u16, Tx>(Option<LocalTxSubmissionClient<ERA, Tx>>);

impl<const ERA: u16, Tx> Drop for ClosingClient<ERA, Tx> {
    fn drop(&mut self) {
        if let Some(client) = self.0.take() {
            // `close` only aborts the two plexer tasks; it completes on its
            // first poll and needs no runtime.
            let _ = futures::FutureExt::now_or_never(client.close());
        }
    }
}

impl<const ERA: u16, Tx> SubmitClient<Tx> for ClosingClient<ERA, Tx>
where
    Tx: Serialize,
{
    async fn submit(&mut self, tx: Tx) -> Result<Response<TxValidationError>, Error> {
        self.0
            .as_mut()
            .expect("the client is taken only on drop")
            .submit_tx(tx)
            .await
    }
}

pub fn tx_submission_agent_stream<'a, const ERA: u16, Tx, Tracker>(
    agent: TxSubmissionAgent<ERA, Tx, Tracker>,
) -> impl Stream<Item = ()> + 'a
where
    Tx: CanonicalHash + Serialize + Clone + 'a,
    Tx::Hash: Display,
    Tracker: TxTracker<Tx::Hash, Tx> + 'a,
{
    let TxSubmissionAgent {
        client,
        mailbox,
        tracker,
        node_config: _,
    } = agent;
    agent_loop(ClosingClient(Some(client)), mailbox, tracker)
}

/// The agent loop. It takes one request at a time from the mailbox. No time
/// limit applies to a submit: the node can hold a local submit until its
/// mempool has room, and the request waits for the node answer. A dropped
/// responder (the caller is gone) is logged, and the loop takes the next
/// request. A protocol error still ends the agent with a panic: its
/// supervisor then opens a new connection.
fn agent_loop<'a, C, Tx, Tracker>(
    mut client: C,
    mut mailbox: mpsc::Receiver<SubmitTx<Tx>>,
    mut tracker: Tracker,
) -> impl Stream<Item = ()> + 'a
where
    C: SubmitClient<Tx> + 'a,
    Tx: CanonicalHash + Clone + 'a,
    Tx::Hash: Display,
    Tracker: TxTracker<Tx::Hash, Tx> + 'a,
{
    stream! {
        loop {
            let SubmitTx(tx, on_resp, hook) = mailbox.select_next_some().await;
            let tx_hash = tx.canonical_hash();
            if let Some(hook) = hook {
                if !hook() {
                    warn!("TX {} not written: its write hook refused the write", tx_hash);
                    continue;
                }
            }
            let submit_result = client.submit(tx.clone()).await;
            match submit_result {
                Ok(Response::Accepted) => {
                    if on_resp.send(SubmissionResult::Ok).is_err() {
                        warn!("TX {} was accepted, but its caller is gone; the answer is not delivered", tx_hash);
                    }
                    tracker.track(tx_hash, tx).await;
                },
                Ok(Response::Rejected(errors)) => {
                    // An unrecognized reason means the pallas error model has
                    // fallen behind the ledger — loud (WARN, with the raw
                    // reason CBOR as hex via Debug) so it is alertable;
                    // ordinary typed rejections keep their quiet trace.
                    match &errors {
                        TxValidationError::Unknown { .. } => {
                            warn!("TX {} was rejected with an unrecognized reason: {:?}", tx_hash, errors)
                        }
                        _ => trace!("TX {} was rejected due to error: {:?}", tx_hash, errors),
                    }
                    if on_resp.send(SubmissionResult::TxRejected{errors: TxRejection(errors)}).is_err() {
                        warn!("TX {} was rejected, but its caller is gone; the answer is not delivered", tx_hash);
                    }
                },
                Err(Error::TxSubmissionProtocol(err)) => {
                    panic!("Failed to submit TX {}: protocol returned error: {}", tx_hash, err);
                },
                Err(err) => panic!("Cannot submit TX {} due to {}", tx_hash, err),
            }
        }
    }
}

impl TryFrom<TxRejection> for HashSet<OutputRef> {
    type Error = &'static str;
    fn try_from(value: TxRejection) -> Result<Self, Self::Error> {
        let mut missing_utxos = HashSet::new();

        if let TxRejection(TxValidationError::ShelleyTxValidationError {
            error: ApplyTxError(node_errors),
            ..
        }) = value
        {
            for error in node_errors {
                if let ConwayLedgerFailure::UtxowFailure(ConwayUtxoWPredFailure::UtxoFailure(
                    UtxoFailure::BadInputsUTxO(inputs),
                )) = error
                {
                    missing_utxos.extend(inputs.into_iter().map(
                        |TransactionInput {
                             transaction_id,
                             index,
                         }| {
                            let tx_hash = *transaction_id;
                            OutputRef::new((*tx_hash).into(), *index)
                        },
                    ));
                }
            }
        }

        if !missing_utxos.is_empty() {
            Ok(missing_utxos)
        } else {
            Err("No missing inputs")
        }
    }
}

#[async_trait::async_trait]
impl<const ERA: u16, Tx> Network<Tx, TxRejection> for TxSubmissionChannel<ERA, Tx>
where
    Tx: Send,
{
    async fn submit_tx(&mut self, tx: Tx) -> Result<(), TxRejection> {
        let (snd, recv) = oneshot::channel();
        self.0.send(SubmitTx(tx, snd, None)).await.unwrap();
        recv.await.expect("Channel closed").into()
    }
}

#[derive(Debug, Clone, derive_more::From)]
pub struct TxRejection(pub TxValidationError);

impl Display for TxRejection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("{:?}", self).as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A tx of the tests: its hash is its number.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestTx(u8);

    impl CanonicalHash for TestTx {
        type Hash = u8;
        fn canonical_hash(&self) -> u8 {
            self.0
        }
    }

    struct NoTracker;

    #[async_trait::async_trait]
    impl TxTracker<u8, TestTx> for NoTracker {
        async fn track(&mut self, _tx_hash: u8, _tx: TestTx) {}
    }

    type Events = Arc<Mutex<Vec<String>>>;

    /// A node connection that accepts every tx and records each write.
    struct ScriptedClient(Events);

    impl SubmitClient<TestTx> for ScriptedClient {
        async fn submit(&mut self, tx: TestTx) -> Result<Response<TxValidationError>, Error> {
            self.0.lock().unwrap().push(format!("write {}", tx.0));
            Ok(Response::Accepted)
        }
    }

    fn hook_logging(events: &Events, tx: u8, write: bool) -> Option<WriteHook> {
        let events = Arc::clone(events);
        Some(Box::new(move || {
            events.lock().unwrap().push(format!("hook {tx}"));
            write
        }))
    }

    fn run_agent(events: &Events) -> mpsc::Sender<SubmitTx<TestTx>> {
        let (snd, recv) = mpsc::channel(8);
        let agent = agent_loop(ScriptedClient(Arc::clone(events)), recv, NoTracker);
        tokio::task::spawn_local(async move {
            futures::pin_mut!(agent);
            while agent.next().await.is_some() {}
        });
        snd
    }

    /// `the-submitter-records-when-a-tx-leaves-for-the-node`: the hook runs
    /// before the client write, and a refusing hook stops the write.
    #[tokio::test]
    async fn the_agent_calls_the_hook_before_the_client_write() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let events: Events = Arc::default();
                let mut mailbox = run_agent(&events);
                let (snd, recv) = oneshot::channel();
                mailbox
                    .send(SubmitTx(TestTx(1), snd, hook_logging(&events, 1, true)))
                    .await
                    .unwrap();
                assert!(matches!(recv.await, Ok(SubmissionResult::Ok)));
                assert_eq!(
                    *events.lock().unwrap(),
                    vec!["hook 1".to_string(), "write 1".to_string()]
                );

                // A refusing hook: no write, and the responder is dropped.
                let (snd, recv) = oneshot::channel();
                mailbox
                    .send(SubmitTx(TestTx(2), snd, hook_logging(&events, 2, false)))
                    .await
                    .unwrap();
                assert!(recv.await.is_err(), "no answer for a refused write");
                assert_eq!(
                    *events.lock().unwrap(),
                    vec!["hook 1".to_string(), "write 1".to_string(), "hook 2".to_string()],
                    "a refused write never reaches the client"
                );
            })
            .await;
    }

    /// `a-held-submit-stays-open-until-the-node-answers`: the caller of the
    /// first request is gone when the node answers. The agent logs, does not
    /// panic, and answers the next request.
    #[tokio::test]
    async fn the_agent_survives_a_dropped_responder() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let events: Events = Arc::default();
                let mut mailbox = run_agent(&events);
                let (snd, recv) = oneshot::channel();
                drop(recv);
                mailbox.send(SubmitTx(TestTx(1), snd, None)).await.unwrap();
                let (snd, recv) = oneshot::channel();
                mailbox.send(SubmitTx(TestTx(2), snd, None)).await.unwrap();
                assert!(
                    matches!(recv.await, Ok(SubmissionResult::Ok)),
                    "the agent took the next request"
                );
                assert_eq!(
                    *events.lock().unwrap(),
                    vec!["write 1".to_string(), "write 2".to_string()]
                );
            })
            .await;
    }

    /// A node connection that holds every write for `hold`, then accepts.
    struct HoldingClient(std::time::Duration);

    impl SubmitClient<TestTx> for HoldingClient {
        async fn submit(&mut self, _tx: TestTx) -> Result<Response<TxValidationError>, Error> {
            tokio::time::sleep(self.0).await;
            Ok(Response::Accepted)
        }
    }

    /// `a-held-submit-stays-open-until-the-node-answers`: no fixed time
    /// limit. The node holds the write past the old 5 s limit; the agent
    /// waits, does not panic, and delivers the answer. Real time: this crate
    /// has no tokio `test-util` for a paused clock.
    #[tokio::test]
    async fn the_agent_waits_for_a_held_write() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let (mut mailbox, recv) = mpsc::channel(8);
                let agent = agent_loop(
                    HoldingClient(std::time::Duration::from_millis(5_500)),
                    recv,
                    NoTracker,
                );
                tokio::task::spawn_local(async move {
                    futures::pin_mut!(agent);
                    while agent.next().await.is_some() {}
                });
                let started = tokio::time::Instant::now();
                let (snd, recv) = oneshot::channel();
                mailbox.send(SubmitTx(TestTx(1), snd, None)).await.unwrap();
                assert!(matches!(recv.await, Ok(SubmissionResult::Ok)));
                assert!(started.elapsed() >= std::time::Duration::from_millis(5_500));
            })
            .await;
    }
}

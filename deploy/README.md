# deploy

Home of the self-contained local-Docker deployment of the MLOps stack
(ADR-0001): the Compose file and container images for MLflow (seeded from the
committed champion artifact), the real-time API, the batch scorer, Prefect
(retraining flow + stream simulator), and monitoring.

Built out in ticket 09, where `make demo` brings the whole stack up from the
committed seed, offline — no cloud, no registry auth, no re-training on a fresh
clone.

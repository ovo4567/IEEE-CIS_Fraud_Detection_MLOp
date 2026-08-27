# Local Docker + no cloud

The deployment stack runs entirely locally via Docker Compose — MLflow, FastAPI, batch scorer, Prefect, and the drift monitor — with no managed cloud services. CI/CD publishes the serving image to GHCR, but the demo builds and runs locally. Chosen over AWS/GCP/Azure for a portfolio/learning project: zero cost, fully reproducible from the repo, and a reviewer can run it without cloud accounts. Consequence: a future cloud migration (e.g. SageMaker/Vertex) is a real lift, not a config change.

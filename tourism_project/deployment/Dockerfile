# Same Python minor version as the GitHub Actions training job, so the pickled
# pipeline is loaded by the interpreter/library versions that created it.
FROM python:3.11

# Hugging Face Spaces run containers as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Work inside the user's home directory so the app can write its model cache.
WORKDIR $HOME/app

# Install dependencies first: this layer is cached until requirements.txt changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the deployment files (app.py, README.md).
COPY --chown=user . .

# Port served by Streamlit; it must match `app_port` in the Space README.md.
EXPOSE 8501

# Launch the Streamlit app and make it reachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

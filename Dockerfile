FROM python:3.11

# Create a non-root user and grant permissions to the workdir.
RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --create-home user

WORKDIR /user/code

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies required by OpenCV.
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

# Copy only the requirements file first to leverage Docker's layer caching.
COPY --chown=user:user requirements.txt .

RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install -r requirements.txt --no-cache-dir

# Copy the rest of the application code.
COPY --chown=user:user . .

# Switch to the non-root user.
USER user

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "7860"]

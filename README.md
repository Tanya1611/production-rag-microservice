# RAG Q&A System

A **RAG (Retrieval-Augmented Generation)** system that enables intelligent Q&A over provided documents with FastAPI & LangChain.

---

## Overview

This project explores how RAG can be used to:

1. Load documents
2. Split documents into smaller chunks
3. Create embeddings
4. Store and search document vectors
5. Retrieve relevant information
6. Generate answers using an LLM

---

## Tech Stack

- Python
- LangChain
- OpenAI
- Vector Database
- FastAPI

---

## Basic Workflow

                ┌───────────────┐
                │    Document   │
                └───────┬───────┘
                        │
                        ▼
                ┌────────────────┐
                │ Text Splitting │
                └───────┬────────┘
                        │
                        ▼
                ┌───────────────┐
                │   Embeddings  │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │  Vector Store │
                └───────┬───────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Similarity Search │
              └─────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Relevant Context │
               └────────┬─────────┘
                        │
                        ▼
                ┌───────────────┐
                │      LLM      │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    Response   │
                └───────────────┘

---

## Current Status

This is the initial version of the project.

Future improvements may include:

* Better document processing
* API endpoints
* Evaluation
* Testing
* Docker
* Deployment
* Observability 
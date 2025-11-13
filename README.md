# multisector-pyFi
A multi-sector ingestion, analysis and financial strategy engine

Use the docker-compose to start a demo PostgreSQL database to interact with
From the root folder:
  1. start the container: docker compose up -d
       container: postgres_demo
       user: postgres
       pwd: pwd
       database: postgres
  2. use client.py to interact with the database
  3. check database schema and tables: docker exec -it postgres_demo psql -U postgres -d postgres then \dt

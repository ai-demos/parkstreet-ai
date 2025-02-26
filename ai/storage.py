from agno.storage.agent.postgres import PostgresAgentStorage

from db.session import db_url

sql_agent_storage = PostgresAgentStorage(
    schema="ai",
    db_url=db_url,
    table_name="sql_agent_storage",
)

analytics_ai_storage = PostgresAgentStorage(
    schema="ai",
    db_url=db_url,
    table_name="analytics_ai_runs",
)

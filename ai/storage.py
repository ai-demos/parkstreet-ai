from phi.storage.agent.postgres import PgAgentStorage

from db.session import db_url

sql_agent_storage = PgAgentStorage(
    schema="ai",
    db_url=db_url,
    table_name="sql_agent_storage",
)

analytics_ai_storage = PgAgentStorage(
    schema="ai",
    db_url=db_url,
    table_name="analytics_ai_runs",
)

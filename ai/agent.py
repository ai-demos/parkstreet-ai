import json
from enum import Enum
from textwrap import dedent
from typing import Optional, List, Union

from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from pydantic import BaseModel, Field

from ai.storage import sql_agent_storage, analytics_ai_storage
from ai.settings import agent_settings
from utils.log import logger


class Table(str, Enum):
    suppliers = "shipments"


class SortOrder(str, Enum):
    asc = "ASC"
    desc = "DESC"


class OrderByColumn(BaseModel):
    column_name: str
    sort_order: SortOrder


class DynamicValue(BaseModel):
    column_name: str


class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="


class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]


class SqlQuery(BaseModel):
    table_name: Table = Field(..., description="The name of the table to query.")
    columns: List[str] = Field(..., description="The columns to include in the SELECT clause.")
    conditions: List[Condition] = Field(..., description="The conditions to include in the WHERE clause.")
    order_by_columns: List[OrderByColumn] = Field(
        ..., description="The columns to order by, including the sort order."
    )
    group_by_columns: List[str] = Field(..., description="The columns to group by in the query.")


semantic_model = {
    "tables": {
        "shipments": {
            "description": "Tracks and manages the lifecycle of shipments, including key events, compliance, and tracking details.",
            "columns": {
                "shipment_id": {
                    "type": "bigint",
                    "null": False,
                    "description": "Unique identifier for the shipment, typically used for tracking",
                },
                "status": {
                    "type": "integer",
                    "description": "Current status of the shipment",
                },
                "departure_date": {
                    "type": "datetime",
                    "description": "Scheduled departure date for the shipment",
                },
                "arrival_date": {
                    "type": "datetime",
                    "description": "Scheduled or actual arrival date of the shipment",
                },
                "cleared_customs": {
                    "type": "datetime",
                    "description": "Date and time when the shipment cleared customs",
                },
                "pol": {
                    "type": "string",
                    "description": "Port of loading for the shipment",
                },
                "pod": {
                    "type": "string",
                    "description": "Port of discharge for the shipment",
                },
                "entry_number": {
                    "type": "string",
                    "description": "Customs entry number assigned to the shipment",
                },
                "client_id": {
                    "type": "bigint",
                    "null": False,
                    "description": "Identifier for the client associated with the shipment",
                },
                "account_manager": {
                    "type": "bigint",
                    "description": "Identifier for the account manager handling the shipment",
                },
                "container_number": {
                    "type": "string",
                    "description": "Unique identifier for the container used in the shipment",
                },
                "cargo_type": {
                    "type": "string",
                    "description": "Type of cargo in the shipment",
                },
                "quoted_rate": {
                    "type": "decimal",
                    "description": "Quoted rate for the shipment",
                },
                "date_added": {
                    "type": "datetime",
                    "precision": 6,
                    "null": False,
                    "description": "Timestamp when the shipment record was added to the system",
                },
                "last_updated": {
                    "type": "datetime",
                    "precision": 6,
                    "null": False,
                    "description": "Timestamp when the shipment record was last updated",
                },
            },
        },
    },
}


def format_condition(condition: Condition) -> str:
    """
    Formats a single condition into a SQL condition string.

    Args:
        condition (Condition): Condition object containing column, operator, and value

    Returns:
        str: Formatted SQL condition string
    """
    # Handle column name
    column = f'"{condition.column}"' if condition.column.lower() == "name" else condition.column

    # Handle value based on its type
    if isinstance(condition.value, DynamicValue):
        value = condition.value.column_name
    elif isinstance(condition.value, str):
        value = f"'{condition.value}'"
    else:  # for integers or other numeric types
        value = str(condition.value)

    return f"{column} {condition.operator.value} {value}"


def format_order_by_column(order_by: OrderByColumn) -> str:
    """
    Formats a single order by column into a SQL ORDER BY clause string.

    Args:
        order_by (OrderByColumn): OrderByColumn object containing column_name and sort_order

    Returns:
        str: Formatted SQL ORDER BY clause string
    """
    # Handle column name
    column = f'"{order_by.column_name}"' if order_by.column_name.lower() == "name" else order_by.column_name

    return f"{column} {order_by.sort_order.value}"


def get_sql_agent(
    team_id: str,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    instructions = [
        "1. Analyze the Question and the expected answer:"
        "   - Based on that the question and the expected answer, identify the table(s) and column(s) that need to be queried.",
        "   - Refer to the semantic model for the available tables and columns.",
        "2. Build the SQL Query:",
        "   - Build the query using the `SqlQuery` model.",
        "   - Make sure to follow the syntax for PostgreSQL for the query.",
        "   - Always include necessary conditions in the WHERE clause, such as:",
        "       - Date ranges with both lower and upper bounds (e.g., `date_column >= NOW()` and `date_column <= NOW() + INTERVAL 'N days'`).",
        "       - `IS NOT NULL` checks for relevant columns to ensure they have valid values.",
        "       - Exclude past records when the question implies future dates by adding `date_column >= NOW()` to the WHERE clause.\n",
        "## General Guidelines:",
        "   - Safe Querying: Be cautious to prevent syntax errors and ensure the generated SQL is valid.",
        "   - Clarity: Ensure your responses are clear and directly address the user's question.",
        "   - Make sure to query boolean columns correctly according to the semantic model (e.g., `diverse = TRUE` or `diverse = FALSE` and not `diverse = 1` or `diverse = 0`).",
        "   - Avoid Including Expired or Irrelevant Records:",
        "   - Carefully specify date conditions and status filters in the WHERE clause.",
        "   - Double-check the logic to prevent unintended inclusions.",
        "   - Always make sure the columns you are querying are available in the table you are querying by referring to the semantic model.",
        "",
    ]

    sql_agent = Agent(
        name="SQLAgent",
        description="You are a PostgreSQL Agent and your task is to answer questions using SQL queries.",
        run_id=run_id,
        user_id=user_id,
        model=OpenAIChat(
            id=agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        additional_context=dedent(f"""\
Here are the tables available:
<semantic_model>
{json.dumps(semantic_model, indent=4)}
</semantic_model>
"""),
        response_model=SqlQuery,
        structured_outputs=True,
        storage=sql_agent_storage,
        monitoring=True,
        debug_mode=debug_mode,
        instructions=instructions,
        user_data={"team_id": team_id},
    )
    return sql_agent


def get_analytics_agent(
    team_id: str,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    def get_answer_using_sql(
        question: str,
        expected_answer: str,
    ) -> str:
        """
        Use this tool to answer a question by running a SQL query.

        Args:
            question (str): The question to answer by running a SQL query.
            expected_answer (str): The expected answer to the question, help the model write the SQL query accurately.

        Returns:
            str: Result of the SQL query or an error message if the query fails.
        """
        logger.info(f"Question: {question}")
        logger.info(f"Expected Answer: {expected_answer}")

        try:
            sql_agent: Agent = get_sql_agent(team_id, run_id, user_id, debug_mode)

            result = None
            previous_query = None
            num_tries = 0
            max_tries = 3
            error_message = None

            while result is None and num_tries < max_tries:
                num_tries += 1

                prompt = f"Write a query to answer the following question: {question}\nExpected Answer: {expected_answer}"
                if error_message is not None and previous_query is not None:
                    prompt += "\n\n"
                    prompt += "The previous query was not correct, please try again."
                    prompt += f"\nThe previous query was: {previous_query}"
                    prompt += "\n"
                    prompt += "The error message was: "
                    prompt += f"{error_message}"
                    prompt += "\n"
                    prompt += "Please correct the query and try again."

                try:
                    sql_agent_response: RunResponse = sql_agent.run(prompt)
                    sql_query: SqlQuery = sql_agent_response.content  # type: ignore
                    logger.info(f"SqlQuery: {sql_query}")
                    previous_query = sql_query
                except Exception as e:
                    error_message = str(e)
                    continue

                query_limit = 10

                formatted_columns = []
                for col in sql_query.columns:
                    if col == "name":
                        formatted_columns.append('"name"')
                    else:
                        formatted_columns.append(col)
                columns_clause = ", ".join(formatted_columns) if formatted_columns else "*"

                query = f"SELECT {columns_clause} FROM {sql_query.table_name.value}"

                if sql_query.conditions:
                    formatted_conditions = [format_condition(condition) for condition in sql_query.conditions]
                    query += f" WHERE {' AND '.join(formatted_conditions)}"
                    query += f" AND organization_id = {team_id}"

                else:
                    query += f" WHERE organization_id = {team_id}"

                if sql_query.group_by_columns:
                    formatted_group_by_columns = [
                        f'"{column}"' if column.lower() == "name" else column
                        for column in sql_query.group_by_columns
                    ]
                    query += f" GROUP BY {', '.join(formatted_group_by_columns)}"

                if sql_query.order_by_columns:
                    formatted_order_by_columns = [
                        format_order_by_column(order_by_column)
                        for order_by_column in sql_query.order_by_columns
                    ]
                    query += f" ORDER BY {', '.join(formatted_order_by_columns)}"

                query += f" LIMIT {query_limit}"

                try:
                    # result = run_query(query)
                    result = query
                except Exception as e:
                    logger.error(f"Error running query: {e}")
                    error_message = None
                    continue

            if result is None:
                error_message = "I couldn't find the answer to your question, please try again."

            return result  # type: ignore

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            return f"Validation error: {ve}"
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return "Error executing query. Please try again later."

    instructions: List[str] = [
        "First **think** about the users question and categorize the question into:\n"
        + "  - General Parkstreet question: Questions about using the Parkstreet platform\n"
        + "  - Sql Question: Questions that can be answered using a SQL query",
        "If the users question is not related to Parkstreet or procurement information, politely redirect the user back to their Parkstreet account and do not answer the question.",
        "If you have all the information you need to answer the users question, provide the answer.",
        "Otherwise answer each category of question using:\n"
        + "  - General Parkstreet related question: Call the `search_knowledge_base` function and answer using the response.\n"
        + "  - Sql Question: call the `get_answer_using_sql` function and answer using the response. Make sure to clearly explain the question and the expected answer.",
        "Guidelines:\n"
        + "  - Do not mention that you are using SQL or your knowledge base to answer the question, just say you are retrieving the necessary information.",
    ]

    analytics_agent = Agent(
        name=f"AnalyticsAIv2_{team_id}" if team_id else "AnalyticsAI",
        run_id=run_id,
        user_id=user_id,
        model=OpenAIChat(
            id=agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        storage=analytics_ai_storage,
        # knowledge=analytics_ai_knowledge,
        debug_mode=debug_mode,
        search_knowledge=True,
        read_chat_history=True,
        description="You are an AI Agent called Parkstreet-AI. You can answer questions related to analytics using SQL queries.",
        instructions=instructions,
        user_data={"team_id": team_id},
        tools=[get_answer_using_sql],
    )
    return analytics_agent

import json
from enum import Enum
from textwrap import dedent
from typing import Optional, List, Union

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

from ai.storage import sql_agent_storage, analytics_ai_storage
from ai.settings import agent_settings
from utils.log import logger


class Table(str, Enum):
    shipments = "shipments"
    orders = "orders"
    financial_snapshots = "financial_snapshots"


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
                "supplier": {
                    "type": "string",
                    "description": "Supplier of the goods in the shipment",
                },
            },
        },
        "orders": {
            "description": "Tracks and manages data about Sales Orders and Transfer Orders, detailing their lifecycle and system information.",
            "columns": {
                "id": {
                    "type": "integer",
                    "null": False,
                    "description": "Primary key, unique for each line item.",
                },
                "transaction_type": {
                    "type": "string",
                    "description": 'Defines order type ("Sales Order" or "Transfer Order").',
                },
                "po_number": {
                    "type": "string",
                    "description": "Purchase Order number for tracking.",
                },
                "client_id": {
                    "type": "bigint",
                    "description": "ID of the related client, based on the CRM classes table.",
                },
                "customer_id": {
                    "type": "bigint",
                    "description": "ID of the related customer, based on the CRM customers table.",
                },
                "so_number": {
                    "type": "string",
                    "description": "Sequential Sales Order number.",
                },
                "approval_status": {
                    "type": "string",
                    "description": "Indicates whether an order is approved.",
                },
                "so_status": {
                    "type": "string",
                    "description": "Current order status in the system (e.g., Fulfilled, Pending Receipt).",
                },
                "delivery_date": {
                    "type": "datetime",
                    "description": "Expected delivery date of the order.",
                },
                "special_instructions": {
                    "type": "string",
                    "description": "Special instructions for order handling.",
                },
                "posted_date": {
                    "type": "datetime",
                    "description": "Date and time when the order was created.",
                },
                "updated_at": {
                    "type": "datetime",
                    "description": "Last time the order record was updated.",
                },
                "delivery_start_datetime": {
                    "type": "datetime",
                    "description": "Start time for delivery.",
                },
                "delivery_end_datetime": {
                    "type": "datetime",
                    "description": "End time for delivery.",
                },
                "freight": {
                    "type": "string",
                    "description": "Shipping terms (e.g., Prepaid or Collect).",
                },
                "carrier": {
                    "type": "string",
                    "description": "Name of the carrier handling the order.",
                },
                "delivery_cost": {
                    "type": "decimal",
                    "description": "Cost of delivery.",
                },
                "fuel_surcharge": {
                    "type": "decimal",
                    "description": "Fuel surcharge applied to the delivery.",
                },
                "fb_status": {
                    "type": "string",
                    "description": "Legacy NetSuite integration order status.",
                },
                "sync_to_warehouse": {
                    "type": "boolean",
                    "description": "Indicates whether the order syncs to a warehouse.",
                },
                "credit_review_status": {
                    "type": "string",
                    "description": "Status of the credit review process.",
                },
                "credit_review_score": {
                    "type": "integer",
                    "description": "Credit review score for the order.",
                },
                "order_group_id": {
                    "type": "integer",
                    "description": "Groups chained orders together.",
                },
                "order_sequence": {
                    "type": "integer",
                    "description": "Sequence order for chained orders.",
                },
            },
        },
        "financial_snapshots": {
            "description": "Analyzes financial snapshots for clients, tracking cash balances, accounts receivable/payable, inventory levels, and credit risk to understand their financial health and operational status over time.",
            "columns": {
                "id": {
                    "type": "integer",
                    "null": False,
                    "description": "Primary key, unique for each snapshot entry.",
                },
                "client_id": {
                    "type": "bigint",
                    "null": False,
                    "description": "Unique identifier for the client.",
                },
                "accounts_receivable": {
                    "type": "decimal",
                    "description": "Total amount of money owed to the client by their customers.",
                },
                "accounts_payable": {
                    "type": "decimal",
                    "description": "Total amount of money the client owes to their suppliers.",
                },
                "cash_balance": {
                    "type": "decimal",
                    "description": "Current cash balance of the client.",
                },
                "net_inventory_value": {
                    "type": "decimal",
                    "description": "Value of the client's inventory, net of costs and depreciation.",
                },
                "total_net_assets": {
                    "type": "decimal",
                    "description": "Total value of the client's net assets.",
                },
                "last_month_sales": {
                    "type": "decimal",
                    "description": "Sales made by the client in the last month.",
                },
                "last_month_deposits": {
                    "type": "decimal",
                    "description": "Total deposits made by the client in the last month.",
                },
                "available_inventory_qty": {
                    "type": "integer",
                    "description": "Quantity of inventory available for sale.",
                },
                "inventory_on_water": {
                    "type": "integer",
                    "description": "Inventory currently in transit or on order.",
                },
                "last_updated": {
                    "type": "datetime",
                    "description": "Date and time when the snapshot was last updated.",
                },
                "last_deposit_date": {
                    "type": "datetime",
                    "description": "Date of the most recent deposit.",
                },
                "credit_risk": {
                    "type": "string",
                    "description": "Credit risk level assigned to the client.",
                },
                "credit_risk_last_change": {
                    "type": "datetime",
                    "description": "Date when the credit risk status was last changed.",
                },
                "cash_balance_notification": {
                    "type": "boolean",
                    "description": "Indicates whether notifications are enabled for cash balance thresholds.",
                },
                "credit_risk_notification": {
                    "type": "boolean",
                    "description": "Indicates whether notifications are enabled for credit risk changes.",
                },
                "status_id": {
                    "type": "integer",
                    "description": "Identifier for the current status of the client or snapshot.",
                },
                "client_type": {
                    "type": "tinyint",
                    "description": "Type of client, categorized by an integer value.",
                },
                "relation_manager_id": {
                    "type": "bigint",
                    "description": "Identifier for the relationship manager assigned to the client.",
                },
            },
        },
    },
}


def run_query(query: str) -> str:
    """
    Runs a SQL query and returns the result.

    Args:
        query (str): The SQL query to run.

    Returns:
        str: The result of the SQL query.
    """

    from db.session import db_url
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query)).fetchall()

        return f"Result: {result} obtained from the query: {query}"

    except Exception as e:
        return f"Error: {e} occurred while running the query: {query}"


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
    team_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
        "   - Only include the columns in the SQL query which are available to you in the semantic model. Do not make up columns names as that would result in an error.",
        "",
    ]

    sql_agent = Agent(
        name="SQLAgent",
        description="You are a PostgreSQL Agent and your task is to answer questions using SQL queries.",
        session_id=session_id,
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
        debug_mode=debug_mode,
        instructions=instructions,
    )
    return sql_agent


def get_analytics_agent(
    team_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
            sql_agent: Agent = get_sql_agent(session_id=session_id, user_id=user_id, debug_mode=debug_mode)

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
                    # query += f" AND organization_id = {team_id}"

                # else:
                # query += f" WHERE organization_id = {team_id}"

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
                    result = run_query(query)
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
        "Your task is to answer the users questions by making a tool call to the `get_answer_using_sql` function.",
        "If the user asks a specific question, make the tool call with the question and the expected answer.",
        # "If the users question is not related to Parkstreet, politely redirect the user back to their Parkstreet account and do not answer the question.",
        # "If you have all the information you need to answer the users question, provide the answer.",
        # "Otherwise answer each category of question using:\n"
        "  - Sql Question: call the `get_answer_using_sql` function and answer using the response. Make sure to clearly explain the question and the expected answer.",
        "Guidelines:\n",
        "  - Do not mention that you are using SQL or your knowledge base to answer the question, just say you are retrieving the necessary information.",
    ]

    analytics_agent = Agent(
        name=f"ParkstreetAI_{team_id}" if team_id else "ParkstreetAI",
        user_id=user_id,
        session_id=session_id,
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
        tools=[get_answer_using_sql],
        add_history_to_messages=True,
        num_history_responses=2,
    )
    return analytics_agent

"""Utilities for generating agent representations."""

import inspect
from typing import Any


def generate_agent_representation(agent_object: Any, agent_type: str) -> str:
    """
    Generate a string representation of an agent based on its type.

    Args:
        agent_object: The agent object to represent
        agent_type: The type of agent (e.g., "langgraph")

    Returns:
        String representation of the agent

    Raises:
        ValueError: If agent type is not supported
    """
    if agent_type.lower() == "langgraph":
        return _generate_langgraph_representation(agent_object)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def _generate_langgraph_representation(workflow_graph: Any) -> str:
    """
    Generate representation for LangGraph workflow.

    Args:
        workflow_graph: LangGraph workflow object

    Returns:
        Formatted string representation
    """
    # Format nodes in the workflow graph
    nodes_formatted = ""
    for node_name, node in workflow_graph.nodes.items():
        # unwrap the original function from the RunnableLambda
        original_func = (
            node.runnable.func if hasattr(node.runnable, "func") else node.runnable
        )
        doc = inspect.getdoc(original_func)
        nodes_formatted += f"- Node: `{node_name}`\nDescription: {doc}\n\n"

    edges_formatted = "\n".join(
        [
            f"`{starting_node}` -> `{ending_node}`"
            for starting_node, ending_node in workflow_graph.edges
        ]
    )

    branches_formatted = ""
    for node, branches in workflow_graph.branches.items():
        for branch_name, branch_object in branches.items():
            branches_formatted += f"""Branches for node `{node}`:
`{branch_name}` -> `{str(branch_object.ends)}`\n"""

    workflow_formatted = f"""# Nodes
{nodes_formatted}

# Edges
{edges_formatted}

# Branches
{branches_formatted}"""
    return workflow_formatted

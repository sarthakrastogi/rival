from rival_ai import (
    AgentDefinition,
    TestCaseGenerator,
    Benchmarking,
)

# Create agent definition for a LangGraph agent graph.
agent_definition = AgentDefinition(
    name="CustomerSupportAgent",
    description="An agent that provides customer support by answering questions and resolving issues.",
    agent_object=customer_support_graph,
    agent_type="langgraph",
    input_schema={"customer_input": "string"},
    output_schema={"reply": "string"},
)

# Generate test cases
generator = TestCaseGenerator(model="gpt-4.1-nano")
my_benchmarking = Benchmarking(project_id="customer_support_agent_0.1.0")

# Process each test case
for i, testcase in enumerate(
    generator.generate_local(
        agent_definition=agent_definition,
    )
):
    # Run your agent with the test case inputs
    agent_response = process_customer_message(inputs=testcase.inputs)

    # Evaluate the agent's response against the test case
    eval_result = testcase.evaluate(
        project_id=project_id,
        agent_definition=agent_definition,
        agent_response=agent_response,
    )

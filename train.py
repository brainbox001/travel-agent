from agent import agent
import asyncio

async def train():

    await agent.ingest("faqs_and_policies.txt")
    await agent.ingest("company_travel_policy.txt")
    await agent.ingest("airline_policies.txt")


asyncio.run(train())

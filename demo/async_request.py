import asyncio
import aiohttp

    
async def do_post(url, instruction, description):
    global data
    data = []
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'instruction': instruction, 'description': description}) as response:
            data.append(await response.json())


def main(url, instruction, description):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        asyncio.gather(
            *(do_post(u, instruction, description) for u in url)
        )
    )
    return data

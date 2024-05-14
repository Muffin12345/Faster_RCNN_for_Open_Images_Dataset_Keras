import asyncio
import random as rnd

async def randomTask(task):
    randomTimeTaken = rnd.randint(1, 10)
    print("Task ", task, " will take: ", str(randomTimeTaken), " seconds!")
    await asyncio.sleep(randomTimeTaken)
    print("Task ", task, " is done!")

async def main():
    await asyncio.gather(randomTask(1), randomTask(2), randomTask(3))

asyncio.run(main())
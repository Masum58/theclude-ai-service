# WHAT: Redis worker process that consumes queued video-processing jobs
# WHY: RQ's SpawnWorker (multiprocessing-based) crashes on Windows because it internally
#      calls os.wait4(), which doesn't exist on this platform. SimpleWorker runs jobs in
#      the same process (no child process spawning), so it works correctly on Windows.
# INPUT: none — reads REDIS_URL and VIDEO_QUEUE_NAME from settings
# OUTPUT: none — runs forever, picking up and executing jobs from the queue
# WHO CALLS: run manually (python -m app.worker) in a dedicated terminal
# CONNECTED TO: Redis, app.tasks.video_processing

from redis import Redis
from rq import SimpleWorker
from rq.timeouts import TimerDeathPenalty

from app.core.config import settings


class WindowsSimpleWorker(SimpleWorker):
    """
    Windows-compatible RQ worker.

    SimpleWorker avoids spawning a child process entirely (so no os.wait4() call is
    ever made — that's what breaks SpawnWorker on Windows).
    TimerDeathPenalty avoids Unix SIGALRM-based timeout handling, which Windows also
    doesn't support.
    """

    death_penalty_class = TimerDeathPenalty


def main() -> None:
    redis_connection = Redis.from_url(settings.REDIS_URL)

    worker = WindowsSimpleWorker(
        queues=[settings.VIDEO_QUEUE_NAME],
        connection=redis_connection,
    )

    print(f"Starting worker for queue: {settings.VIDEO_QUEUE_NAME}")

    worker.work()


if __name__ == "__main__":
    main()
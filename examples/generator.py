import math
from random import Random
from math import pow, log, exp

from model.linear_system import Processor, LinearSystem, SchedulerType, Flow, Task


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum_u = utilization
    us = []

    for i in range(1, n_tasks):
        next_sum_u = sum_u * pow(random.random(), 1 / (n_tasks - i))
        us.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    us.append(sum_u)
    return us


def log_uniform(random: Random, lowest: float, highest: float) -> float:
    r = random.uniform(log(lowest), log(highest))
    return exp(r)


def set_processor_utilization(processor: Processor, utilization: float):
    tasks = processor.tasks
    if len(tasks) == 0:
        return
    factor = utilization / processor.utilization
    for task in tasks:
        task.wcet *= factor


def set_utilization(system: LinearSystem, utilization: float):
    """Sets the given system utilization, by setting the utilization of every processor to the same value"""
    for proc in system.processors:
        set_processor_utilization(proc, utilization)


def set_system_utilization(system: LinearSystem, utilization: float):
    """Sets the WCET's of the tasks such that the sum of all tasks utilization divided by the number of processors
       is equal to 'utilization'"""
    tasks = system.tasks
    u = sum([task.utilization for task in tasks])
    factor = utilization * len(system.processors) / u
    for task in tasks:
        task.wcet *= factor


def generate_system(random: Random, n_flows, n_tasks, n_procs, utilization, sched: SchedulerType,
                    period_min, period_max, deadline_factor_min, deadline_factor_max,
                    balanced=False) -> LinearSystem:
    system = LinearSystem()
    procs = [Processor(name=f"proc{i}", sched=sched) for i in range(n_procs)]
    system.add_procs(*procs)

    # set the general structure
    for f in range(n_flows):
        period = log_uniform(random, period_min, period_max)
        deadline = random.uniform(
            period * n_tasks * deadline_factor_min,
            period * n_tasks * deadline_factor_max)
        flow = Flow(name=f"flow{f}", period=period, deadline=deadline)

        # for now leave the WCET empty
        tasks = [Task(name=f"task{f}_{t}", wcet=0, processor=random.choice(procs)) for t in range(n_tasks)]
        flow.add_tasks(*tasks)
        system.add_flows(flow)

    # if balanced=True, balance the number of tasks per processor (ignore current mapping)
    if balanced:
        # r = Random(len(system.tasks))
        tasks = system.tasks
        random.shuffle(tasks)
        for i, task in enumerate(tasks):
            task.processor = procs[i % len(procs)]

    # set the WCET's
    for proc in procs:
        tasks = proc.tasks
        if tasks:
            us = uunifast(random, len(tasks), utilization)
            for task, u in zip(tasks, us):
                task.wcet = u * task.period

    return system


def unbalance(system: LinearSystem):
    """Heavily unbalance the system by using the least amount of processors possible"""
    bins = []
    bin = []
    u = 0
    for task in system.tasks:
        task.processor = None
        if u + task.utilization < 1:
            bin.append(task)
            u += task.utilization
        else:
            bins.append(bin)
            bin = [task]
            u = task.utilization

    if len(bin) > 0:
        bins.append(bin)

    procs = system.processors
    for i, bin in enumerate(bins):
        proc = procs[i % len(procs)]
        for task in bin:
            task.processor = proc


def to_edf(system: LinearSystem, local=True):
    for proc in system.processors:
        proc.sched = SchedulerType.EDF
        proc.local = local
    return system


def to_int(system: LinearSystem):
    for flow in system:
        flow.period = int(flow.period)
        flow.deadline = int(flow.deadline)
        for task in flow.tasks:
            task.wcet = int(task.wcet)
            if task.wcet == 0:
                task.wcet = 1
            task.deadline = int(task.deadline)
            task.priority = int(task.priority)
    return system


def copy(system: LinearSystem):
    new_procs = {proc.name: Processor(name=proc.name, sched=proc.sched) for proc in system.processors}
    new_system = LinearSystem()

    for flow in system:
        new_flow = Flow(name=flow.name, period=flow.period, deadline=flow.deadline)

        for task in flow:
            new_task = task.copy()
            new_task.processor = new_procs[task.processor.name]
            new_flow.add_tasks(new_task)
        new_system.add_flows(new_flow)

    return new_system


def create_series(template: LinearSystem, utilizations) -> [LinearSystem]:
    systems = []
    for utilization in utilizations:
        system = copy(template)
        for proc in system.processors:
            set_processor_utilization(proc, utilization)
        systems.append(system)
    return systems

# def walk_series(system: System, utilizations, callback) -> None:
#     save_tasks_params(system)
#     for utilization in utilizations:
#         for proc in system.processors:
#             set_processor_utilization(proc, utilization)
#         if callback:
#             callback(system)
#     restore_tasks_params(system)

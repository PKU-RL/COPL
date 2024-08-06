class Scheduler(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep
        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class ConstantScheduler(Scheduler):
    """
    Value remains constant over time.
    :param value: (float) Constant value of the schedule
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


class LinearScheduler(Scheduler):
    """
    Linear interpolation between initial_value and final_value over
    schedule_steps. After this many steps pass final_value is returned.
    """

    def __init__(self, value, schedule_steps, final_value):
        self.schedule_steps = schedule_steps
        self.initial_value = value
        self.final_value = final_value

    def value(self, step):
        fraction = min(float(step) / self.schedule_steps, 1.0)
        return self.initial_value + fraction * (self.final_value - self.initial_value)

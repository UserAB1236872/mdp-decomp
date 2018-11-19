from abc import abstractmethod
import abc


class Decay(abc.ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass


class LinearDecay(Decay):
    def __init__(self, start: float, end: float, steps: int) -> float:
        self.__start = start
        self.__end = end
        self.__steps = steps
        self.__curr_steps = 0
        self.__curr_epsilon = start

    @property
    def start(self) -> float:
        return self.__start

    @property
    def end(self) -> float:
        return self.__start

    @property
    def steps(self) -> int:
        return self.__start

    @property
    def curr_steps(self) -> int:
        return self.__start

    @property
    def curr_epsilon(self) -> float:
        return self.__start

    def __call__(self) -> float:
        self.__curr_steps += 1
        interp = min(self.curr_steps, self.steps) / self.steps
        self.__curr_epsilon = (1 - interp) * self.start + interp * self.end

        return self.curr_epsilon

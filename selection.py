import random

class BinaryTournamentSelection():

    def __init__(self):
        pass

    def execute(self, front: []) -> []:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = random.sample(range(0, len(front)), 2)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result
    
    def compare(self, solution1, solution2) -> int:
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")

        value1 = solution1.objectives[0]
        value2 = solution2.objectives[0]
        result = 0
        if value1 != value2:
            if value1 < value2:
                result = -1
            if value1 > value2:
                result = 1

        return result
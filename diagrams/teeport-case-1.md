graph TD
    O[Optimizer]
    E[Evaluator]

    subgraph Teeport Server
    T[Task]
    end

    O --- T
    E --- T
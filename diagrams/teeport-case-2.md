graph TD
    subgraph Local
    O[Optimizer]
    end

    subgraph Remote
    E[Evaluator]
    end

    subgraph Teeport Server
    T[Task]
    end

    O --- T
    E --- T
graph TD
    subgraph Local
    E[Evaluator]
    end

    subgraph Remote
    O[Optimizer]
    end

    subgraph Teeport Server
    T[Task]
    end

    E --- T
    O --- T
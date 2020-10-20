sequenceDiagram
    Note over Wildcard: Get optimizer/evaluator id
    Wildcard->>+Server: init task
    Server->>Server: create task in memory
    par
        Server->>+Monitor: start task
        Monitor->>-Monitor: create task in memory
    and
        Server->>-Optimizer: start task
    end
    loop optimization
        activate Optimizer
        Optimizer->>Optimizer: generate next X
        Optimizer->>+Server: evaluate X
        deactivate Optimizer
        Server->>Server: add X to pending queue
        Server->>-Evaluator: evaluate X
        activate Evaluator
        Evaluator->>Evaluator: calculate Y = f(X)
        Evaluator->>+Server: evaluated Y
        deactivate Evaluator
        Server->>Server: remove X from pending queue
        Server->>Server: add [X, Y] to history list
        par
            Server->>+Monitor: evaluated Y
            Monitor->>-Monitor: add Y to history list
        and
            Server->>-Optimizer: evaluated Y
        end
    end
    Note over Optimizer: Meet termination condition
    Optimizer->>+Server: task completed
    Server->>Server: dump task to DB
    Server->>-Monitor: task completed
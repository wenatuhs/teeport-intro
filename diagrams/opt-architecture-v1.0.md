graph TD
    subgraph Teeport Clients
    C1[Evaluator]
    C2[Optimizer]
    C3[Processor]
    C4[Monitor/Controller]
    end

    S[[Teeport server]]

    subgraph Storage
    R[(Redis)]
    P[(PostgreSQL)]
    end

    C1 ---|websocket| S
    C2 ---|websocket| S
    C3 ---|websocket| S
    C4 ---|websocket| S

    S ---|in-memory task| R
    S ---|permanentize task| P
sequenceDiagram
    participant H as Human
    participant E as Evaluate Func
    participant O as Optimize Func
    participant S as Socket Thread

    H->>E: first call with X0
    E->>O: create with X0
    par
        E->>E: start eval thread
        activate E
        Note over E: wait for Y0
    and
        O->>+S: evaluate X0
    end
    Note over S: get Y0 from server
    S->>-O: evaluated Y0
    O->>E: set result Y0
    deactivate E
    par
        O->>O: start opt thread
        activate O
        Note over O: wait for X
    and
        E->>H: return Y0
    end

    loop
        H->>E: call with X
        E->>O: set result X
        deactivate O
        par
            E->>+E: start eval thread
            Note over E: wait for Y
        and
            O->>+S: evaluate X
        end
        Note over S: get Y from server
        S->>-O: evaluated Y
        O->>E: set result Y
        deactivate E
        par
            O->>O: start opt thread
            activate O
            Note over O: wait for X
        and
            E->>H: return Y
        end
    end
    deactivate O
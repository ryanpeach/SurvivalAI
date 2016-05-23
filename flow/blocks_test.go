package flow

import (
    "fmt"
    "time"
    "testing"
)

func TestPlus(t *testing.T) {
    // Run a Plus block
    plusblk := Plus(0)
    f_out := make(chan DataOut)
    f_stop := make(chan bool)
    f_err := make(chan FlowError)

    // Run block and put a timeout on the stop channel
    go plusblk.Run(ParamValues{"A": 2, "B": 3}, f_out, f_stop, f_err)
    go timeout(f_stop, 1000)
    
    // Wait for output or error
    switch {
        case out := <-f_out:
        case err := <-f_err:
            if !err.Ok {
                t.Error("Plus.Run returned FlowError: ", err.Info)
            }
    }
    
    // Test the output
    if out != 5 {
        t.Error("Plus.Run returned ", out, " instead of 5.")
    }
}

package flow

import (
    "testing"
)

func TestPlus(t *testing.T) {
    // Run a Plus block
    plusblk := Plus(0)
    f_out := make(chan DataOut)
    f_stop := make(chan bool)
    f_err := make(chan FlowError)
    
    // Test Params
    a, b := float64(2), float64(3)
    c := a + b

    // Run block and put a timeout on the stop channel
    go plusblk.Run(ParamValues{"A": a, "B": b}, f_out, f_stop, f_err)
    go Timeout(f_stop, 100000)
    
    // Wait for output or error
    var out DataOut
    var cont bool = true
    for cont {
        select {
            case out = <-f_out:
                cont = false
            case err := <-f_err:
                if !err.Ok {
                    t.Error("Plus.Run returned FlowError: ", err.Info)
                    return
                }
            case <-f_stop:
                t.Error("Timeout ")
                return
        }
    }
    
    // Test the output
    if out.Values["OUT"] != c {
        t.Error("Plus.Run returned ", out.Values, " instead of ", c, ".")
    }
}

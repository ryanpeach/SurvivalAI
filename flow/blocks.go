package flow

import "time"

func Plus(id InstanceID) FunctionBlock {
    // Create Plus block
    ins := ParamTypes{"A": "float32", "B": "float32"}
    outs := ParamTypes{"OUT": "float32"}
    
    // Define the function as a closure
    runfunc := func(inputs ParamValues,
                     outputs chan DataOut,
                     stop chan bool,
                     err chan FlowError) {
        data := make(ParamValues)
        data["OUT"] = inputs["A"].(float32) + inputs["B"].(float32)
        out := DataOut{BlockID: id, Values: data}
        outputs <- out
        return
    }

    // Initialize the block and return
    outblk := PrimitiveBlock{name: "Plus", fn: runfunc, id: id, inputs: ins, outputs: outs}
    return outblk
}

func Timeout(stop chan bool, sleeptime int) {
    time.Sleep(time.Duration(sleeptime))
    stop <- true
}
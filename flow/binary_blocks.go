package flow

// Creates a variety of blocks for paired operations
func opBinary(id InstanceID, aT,bT,cT TypeStr, outname string, opfunc func(in ParamValues, out *ParamValues)) FunctionBlock {
    // Create Plus block
    ins := ParamTypes{"A": aT, "B": bT}
    outs := ParamTypes{"OUT": cT}
    
    // Define the function as a closure
    runfunc := func(inputs ParamValues,
                     outputs chan DataOut,
                     stop chan bool,
                     err chan FlowError) {
        data := make(ParamValues)
        opfunc(inputs, &data)
        out := DataOut{BlockID: id, Values: data}
        outputs <- out
        return
    }
    
    // Initialize the block and return
    return PrimitiveBlock{name: outname, fn: runfunc, id: id, inputs: ins, outputs: outs}
}

// Numeric Float Functions
func PlusFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) + in["B"].(float64)
    }
    name := "numeric_plus_float"
    return opBinary(id,"float","float","float",name,opfunc)
}
func SubFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) - in["B"].(float64)
    }
    name := "numeric_subtract_float"
    return opBinary(id,"float","float","float",name,opfunc)
}
func MultFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) * in["B"].(float64)
    }
    name := "numeric_multiply_float"
    return opBinary(id,"float","float","float",name,opfunc)
}
func DivFloat(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(float64) / in["B"].(float64)
    }
    name := "numeric_divide_float"
    return opBinary(id,"float","float","float",name,opfunc)
}

// Numeric Int Functions
func PlusInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) + in["B"].(int)
    }
    name := "numeric_plus_int"
    return opBinary(id,"int","int","int",name,opfunc)
}
func SubInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) - in["B"].(int)
    }
    name := "numeric_subtract_int"
    return opBinary(id,"int","int","int",name,opfunc)
}
func MultInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) * in["B"].(int)
    }
    name := "numeric_multiply_int"
    return opBinary(id,"int","int","int",name,opfunc)
}
func DivInt(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) / in["B"].(int)
    }
    name := "numeric_divide_int"
    return opBinary(id,"int","int","int",name,opfunc)
}
func Mod(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(int) % in["B"].(int)
    }
    name := "numeric_mod_int"
    return opBinary(id,"int","int","int",name,opfunc)
}

// Boolean Logic Functions
func And(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) && in["B"].(bool)
    }
    name := "logical_and"
    return opBinary(id,"bool","bool","bool",name,opfunc)
}
func Or(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) || in["B"].(bool)
    }
    name := "logical_or"
    return opBinary(id,"bool","bool","bool",name,opfunc)
}
func Xor(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = in["A"].(bool) != in["B"].(bool)
    }
    name := "logical_xor"
    return opBinary(id,"bool","bool","bool",name,opfunc)
}
func Nand(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = !(in["A"].(bool) && in["B"].(bool))
    }
    name := "logical_nand"
    return opBinary(id,"bool","bool","bool",name,opfunc)
}
func Nor(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = !(in["A"].(bool) || in["B"].(bool))
    }
    name := "logical_nor"
    return opBinary(id,"bool","bool","bool",name,opfunc)
}

// Comparison
func Greater(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) > toNum(in["B"])
    }
    name := "greater_than"
    return opBinary(id,"num","num","bool",name,opfunc)
}
func Lesser(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) < toNum(in["B"])
    }
    name := "lesser_than"
    return opBinary(id,"num","num","bool",name,opfunc)
}
func GreaterEquals(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) >= toNum(in["B"])
    }
    name := "greater_equals"
    return opBinary(id,"num","num","bool",name,opfunc)
}
func LesserEquals(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) <= toNum(in["B"])
    }
    name := "lesser_equals"
    return opBinary(id,"num","num","bool",name,opfunc)
}
func Equals(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) == toNum(in["B"])
    }
    name := "equals"
    return opBinary(id,"num","num","bool",name,opfunc)
}
func NotEquals(id InstanceID) FunctionBlock {
    opfunc := func(in ParamValues, out *ParamValues) {
        (*out)["OUT"] = toNum(in["A"]) != toNum(in["B"])
    }
    name := "not_equals"
    return opBinary(id,"num","num","bool",name,opfunc)
}
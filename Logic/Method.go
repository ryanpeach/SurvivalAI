type Method struct {
    name   string
    params []*Param
    F      interface{}
}

type Address struct {
    block_addr *Block
    param_addr *Param
}

type Flow struct {
    addr_out *Address
    addr_in  *Address
}

type Graph struct {
    all_blocks []*Block
    all_edges  []*Flow
}

func main() {
    a := Block{[]string{"Height", "Width"}, []string{"Region"}}
}
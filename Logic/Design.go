package Logic

import "World3D/Region"

type blockinv struct {
    blocklst Block[]
    quantity uint8[]
}

type Condition Region
func (c Condition) check()

type Implication struct {
    init Condition
    exit Condition
    conn Explanation
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
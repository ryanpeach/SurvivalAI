package Logic

import "World3D/Region"

type blockinv struct {
    blocklst Block[]
    quantity uint8[]
}

type Condition Region
func (c Condition) check() {}

type Implication struct {
    init Condition
    exit Condition
    conn Explanation
}
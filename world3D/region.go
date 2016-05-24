package world3D

type Block struct {
    id uint
    layer uint
    rx, ry, rz uint
}

type RegionType struct {
    space [][][]*Block  // Region's have a 3-dimensional space
    dx, dy, dz uint     // Region's have a size
}

type Region struct {
    R *RegionType
    x, y, z int
}

func (r Region) valid(x, y, z uint) bool {
    x_valid := (x >= 0) && (x <= (r.x + r.R.dx))
    y_valid := (y >= 0) && (y <= (r.y + r.R.dy))
    z_valid := (z >= 0) && (z <= (r.z + r.R.dz))
    return x_valid && y_valid && z_valid
}

func (r1 Region) add(r2 *Region) Region {
    // Find the location
    // Location is the minimum location between the two
    var x0, y0, z0 int
    if r1.x < r2.x {x0 = r1.x} else {x0 = r2.x}
    if r1.y < r2.y {y0 = r1.y} else {y0 = r2.y}
    if r1.z < r2.z {z0 = r1.z} else {z0 = r2.z}
    
    // Find the size
    // Size is the maximum location + it's size in each dimension independently
    var dx, dy, dz uint
    if r1.x + r1.dx > r2.x + r2.dx {dx = r1.x + r1.R.dx} else {dx = r2.x + r2.R.dx}
    if r1.y + r1.dy > r2.y + r2.dy {dy = r1.y + r1.R.dy} else {dy = r2.y + r2.R.dy}
    if r1.z + r1.dz > r2.z + r2.dz {dz = r1.z + r1.R.dz} else {dz = r2.z + r2.R.dz}

    // Create array
    var {
        x, y, z uint                // True pointer space coordinates
        r1i, r1j, r1k uint          // r1 pointer array coordinates
        r2i, r2j, r2k uint          // r2 pointer array coordinates
        r1i_valid, r1j_valid, r1k_valid bool // coordinate validity in r1 dimensions
        r2i_valid, r2j_valid, r2k_valid bool // coordinate validity in r2 dimensions
        r1_valid, r2_valid bool     // coordinate validity in r1 and r2
        out [dx][dy][dz]*Block      // The output space
    }
    
    // Iterate over x
    for i := range out {
        x = i + x0                                    // Find the current x value
        r1i, r2i = x - r1.x, x - r2.x                 // Find the i position in r1 and r2
        r1i_valid = r1i >= 0 && x <= r1.x + r1.R.dx   // Check if x and r1i are valid locations in r1
        r2i_valid = r2i >= 0 && x <= r2.x + r2.R.dx   // Check if x and r2i are valid locations in r2
        
        // Iterate over y
        for j := range out[i] {
            y = j + y0
            r1j, r2j = y - r1.y, y - r2.y
            r1j_valid = r1j >= 0 && y <= r1.y + r1.R.dy
            r2j_valid = r2j >= 0 && y <= r2.y + r2.R.dy
                
            // Iterate over z
            for k := range out[i][j] {
                z = k + z0
                r1k, r2k = z - r1.z, z - r2.z
                r1k_valid = r1k >= 0 && z <= r1.z + r1.R.dz
                r2k_valid = r2k >= 0 && z <= r2.z + r2.R.dz
                
                // Find if coordinates are valid in r1 and r2 independently
                r1_valid = r1i_valid && r1j_valid && r1k_valid
                r2_valid = r2i_valid && r2j_valid && r2k_valid
                
                // Choose the block based on validity and layer
                switch {
                    case r1_valid && r2_valid:  // If they are both valid, choose by layer
                        if r1.R.space[i][j][k].layer > r2.R.space[i][j][k].layer {
                            block = r1.R.space[i][j][k]
                        } else {
                            block = r2.R.space[i][j][k]
                        }
                    case r1_valid:
                        block = r1.R.space[i][j][k]
                    case r2_valid:
                        block = r2.R.space[i][j][k]
                    default:    // If none are valid, an empty block is passed
                        block = &Block{}
                }

                // Set the block at out[coord]
                out[i][j][k] = block
                
            }
        }
    }
    
    // Return a region with space parameter as out, and the size as dx dy dz
    R := RegionType {
        space: out  // Region's have a 3-dimensional space
        dx, dy, dz: dx, dy, dz     // Region's have a size
    }
    return Region {
        R: &R
        x, y, z: x0, y0, z0
    }
}

func (r Region) Draw() {
    return
}
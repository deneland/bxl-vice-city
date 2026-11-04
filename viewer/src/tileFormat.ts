// The voxel tile records written by pipeline/voxelize.py (its constants are in pipeline/common.py):
// 8 bytes per voxel: x+1, y+1, z, r, g, b, LiDAR class, flags (x / y tile-local with a 1-voxel apron).
export const FLAG_FILL = 1;      // geometry synthesised (wall / terrain fill), not measured
export const FLAG_APRON = 2;     // belongs to the neighbouring tile (culling only)
export const FLAG_PHOTO = 8;     // colour from street-level photos
export const C_GROUND = 2;       // LiDAR (ASPRS) ground class

use glam::Vec3;

pub fn collides<F>(pos: Vec3, height: f32, radius: f32, is_solid: &F) -> bool
where
    F: Fn(i32, i32, i32) -> bool,
{
    let min = Vec3::new(pos.x - radius, pos.y, pos.z - radius);
    let max = Vec3::new(pos.x + radius, pos.y + height, pos.z + radius);

    let min_x = min.x.floor() as i32;
    let max_x = (max.x - 0.001).floor() as i32;
    let min_y = min.y.floor() as i32;
    let max_y = (max.y - 0.001).floor() as i32;
    let min_z = min.z.floor() as i32;
    let max_z = (max.z - 0.001).floor() as i32;

    for z in min_z..=max_z {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if is_solid(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}

use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraViewMode {
    FirstPerson,
    ThirdPersonBack,
    ThirdPersonFront,
}

impl CameraViewMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::FirstPerson => Self::ThirdPersonBack,
            Self::ThirdPersonBack => Self::ThirdPersonFront,
            Self::ThirdPersonFront => Self::FirstPerson,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::FirstPerson => "first-person",
            Self::ThirdPersonBack => "third-person",
            Self::ThirdPersonFront => "front camera",
        }
    }

    pub fn shows_player_model(self) -> bool {
        !matches!(self, Self::FirstPerson)
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, self.up)
    }
}

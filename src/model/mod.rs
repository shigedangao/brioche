pub mod fov_model {
    include!(concat!(env!("OUT_DIR"), "/model/depthpro_vit_fov.rs"));
}

pub mod image_model {
    include!(concat!(env!("OUT_DIR"), "/model/depthpro_vit_image.rs"));
}

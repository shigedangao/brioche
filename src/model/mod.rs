pub mod patch {
    include!(concat!(env!("OUT_DIR"), "/model/depthpro_vit_patch.rs"));
}

pub mod image {
    include!(concat!(env!("OUT_DIR"), "/model/depthpro_vit_image.rs"));
}

pub mod fov {
    include!(concat!(env!("OUT_DIR"), "/model/depthpro_vit_fov.rs"));
}

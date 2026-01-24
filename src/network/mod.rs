use crate::brioche_seq::BriocheHeadConfig;
use crate::network::decoder::multires_conv::MultiResDecoderConfig;
use crate::network::encoder::EncoderConfig;
use crate::network::fov::FovConfig;
use anyhow::Result;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

pub mod decoder;
pub mod encoder;
pub mod fov;

pub enum NetworkConfig {
    Encoder(EncoderConfig),
    Fov(FovConfig),
    Decoder(MultiResDecoderConfig),
    Head(BriocheHeadConfig),
}

pub trait Network<B: Backend> {
    fn new(config: NetworkConfig, device: &B::Device) -> Result<Self>
    where
        Self: Sized;
    fn with_record<S>(self, path: S, device: &B::Device) -> Self
    where
        Self: Sized + Module<B>,
        S: AsRef<str>,
    {
        let arg = LoadArgs::new(path.as_ref().into());

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(arg, device)
            .unwrap();

        self.load_record(record)
    }
}

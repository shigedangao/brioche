use crate::brioche_seq::BriocheHeadConfig;
use crate::network::decoder::multires_conv::MultiResDecoderConfig;
use crate::network::encoder::EncoderConfig;
use crate::network::fov::FovConfig;
use anyhow::{Result, anyhow};
use burn::module::Module;
use burn::prelude::Backend;
use burn_store::{ModuleSnapshot, PytorchStore};

pub mod decoder;
pub mod encoder;
pub mod fov;

pub enum NetworkConfig {
    Encoder(EncoderConfig),
    Fov(FovConfig),
    Decoder(MultiResDecoderConfig),
    Head(BriocheHeadConfig),
}

/// Network trait is a trait which defines the basic functionality of a network (encoder, decoder, fov).
pub trait Network<B: Backend> {
    /// Create a new network instance based on the provided configuration and device.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the network.
    /// * `device` - The device on which the network will be created.
    ///
    /// # Returns
    ///
    /// A new network instance.
    fn new(config: NetworkConfig, device: &B::Device) -> Result<Self>
    where
        Self: Sized;

    /// With record load the weight of the model from a pt file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the pt file.
    /// * `device` - The device on which the network will be created.
    ///
    /// # Returns
    ///
    /// A new network instance with the weights loaded from the pt file.
    fn with_record<S>(mut self, path: S) -> Result<Self>
    where
        Self: Sized + Module<B>,
        S: AsRef<str>,
    {
        let mut store = PytorchStore::from_file(path.as_ref());

        self.load_from(&mut store)
            .map_err(|err| anyhow!("Unable to load the model due to {err}"))?;

        Ok(self)
    }
}

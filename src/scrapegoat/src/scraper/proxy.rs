use rand::{rng, seq::IndexedRandom};
use std::{fs::read_to_string, io};

pub struct Proxies {
  proxies: Vec<String>,
}

impl Proxies {
  pub fn new(file: &str) -> Result<Proxies, io::Error> {
    let str_in = read_to_string(file)?;
    let proxies = str_in.lines().map(|s| s.to_string()).collect();

    Ok(Proxies { proxies })
  }

  pub fn get_proxy(&self) -> &String {
    self.proxies.choose(&mut rng()).unwrap()
  }
}

impl Iterator for Proxies {
  type Item = String;

  fn next(&mut self) -> Option<Self::Item> {
    self.proxies.iter().next().cloned()
  }
}

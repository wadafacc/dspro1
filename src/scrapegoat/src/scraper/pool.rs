use reqwest::{Client, Proxy};

use crate::scraper::proxy::Proxies;

const MAX_CONCURRENT: usize = 20;

pub struct ClientPool {
  permits: Permits,
  clients: Vec<Client>,
  current_idx: usize
}

impl ClientPool {
  pub fn new(proxies: Proxies, max_concurrent: usize) -> Self {
    // build a client per proxy
    let clients = proxies
      .map(|p| {
        Client::builder()
          .proxy(Proxy::all(p).unwrap())
          .build()
          .unwrap()
      })
      .collect();

    Self { clients, permits: Permits::new(max_concurrent) }
  }

  pub fn get() -> Result<Client, NoPermitError> {
    todo!()
  }
}

struct Permits {
  issued_permits: usize,
  max_permits: usize,
}

struct NoPermitError {}

impl Permits {
  pub fn new(max_permits: usize) -> Self {
    Self {
      max_permits,
      issued_permits: 0,
    }
  }

  pub fn get(&mut self) -> Result<(), NoPermitError> {
    if self.issued_permits == self.max_permits {
      return Err(NoPermitError {});
    }
    self.issued_permits += 1;
    Ok(())
  }

  pub fn drop(&mut self) -> Result<(), NoPermitError> {
    if self.issued_permits == 0 {
      return Err(NoPermitError {});
    };

    self.issued_permits -= 1;
    Ok(())
  }
}

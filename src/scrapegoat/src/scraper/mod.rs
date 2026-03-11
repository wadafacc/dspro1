use std::io;

use rand::{rng, seq::IndexedRandom};
use reqwest::{Client, ClientBuilder, Proxy};

use crate::scraper::{pool::ClientPool, proxy::Proxies, useragent::UserAgents};

pub mod proxy;
pub mod useragent;
pub mod pool;

trait Scraper {
  fn new(proxy_file: &str, useragents_file: &str) -> Result<ScrapeGoat, io::Error>;
  async fn get_page(&self, url: &str) -> Result<String, reqwest::Error>;
}

pub struct ScrapeGoat {
  user_agents: UserAgents,
  pool: ClientPool, // client list
}

impl Scraper for ScrapeGoat {
  fn new(proxy_file: &str, useragents_file: &str) -> Result<Self, io::Error> {
    let proxies = Proxies::new(proxy_file)?;
    let user_agents = UserAgents::new(useragents_file)?;

    Ok(Self {
      user_agents,
      pool: ClientPool::new(proxies),
    })
  }



  async fn get_page(&self, url: &str) -> Result<String, reqwest::Error> {
    let client = self.clients.choose(&mut rng()).unwrap();

    let res = client.get(url).header("User-Agent", self.user_agents.get_agent()).send().await?;
    Ok(res.text().await.unwrap())
  }
}

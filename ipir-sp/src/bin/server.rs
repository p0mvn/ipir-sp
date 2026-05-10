#[cfg(feature = "http_server")]
use actix_cors::Cors;
#[cfg(feature = "http_server")]
use actix_web::{get, post, web, App, HttpServer};
#[cfg(feature = "http_server")]
use clap::Parser;
#[cfg(feature = "http_server")]
use inspiring::{PackPreprocessed, RlweParams};
#[cfg(feature = "http_server")]
use ipir_sp::client::IPIRClient;
#[cfg(feature = "http_server")]
use ipir_sp::params_for_simplepir;
#[cfg(feature = "http_server")]
use ipir_sp::server::{build_pack_preprocessed_blocks, IPIRServer};

#[cfg(feature = "http_server")]
#[derive(Parser, Debug)]
#[command(version, about = "Run an IPIR-SP HTTP server")]
struct Args {
    /// Number of items in the database.
    num_items: usize,
    /// Size of each item in bits.
    item_size_bits: Option<usize>,
    /// Port.
    #[clap(long, short, default_value = "8080")]
    port: u16,
    /// Deterministic setup seed shared with the demo client.
    #[clap(long, default_value = "7")]
    setup_seed: u64,
}

#[cfg(feature = "http_server")]
struct ServerState {
    rlwe: &'static RlweParams,
    server: IPIRServer<u16>,
    preprocessed: Vec<PackPreprocessed<'static>>,
}

#[cfg(feature = "http_server")]
#[post("/query")]
async fn query(
    body: web::Bytes,
    data: web::Data<ServerState>,
) -> Result<Vec<u8>, actix_web::error::Error> {
    data.server
        .perform_full_online_computation_simplepir(data.rlwe, &body, &data.preprocessed)
        .map_err(actix_web::error::ErrorBadRequest)
}

#[cfg(feature = "http_server")]
#[get("/")]
async fn index(data: web::Data<ServerState>) -> String {
    format!("Hello {}!", data.rlwe.d)
}

#[cfg(feature = "http_server")]
#[get("/info")]
async fn info(data: web::Data<ServerState>) -> String {
    format!(
        "rows={} cols={}",
        data.server.db_rows(),
        data.server.db_cols()
    )
}

#[cfg(feature = "http_server")]
fn seed_from_u64(value: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&value.to_le_bytes());
    seed
}

#[cfg(feature = "http_server")]
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let item_size_bits = args.item_size_bits.unwrap_or(16_384 * 8);
    let (rlwe, ypir) =
        params_for_simplepir(args.num_items as u64, item_size_bits as u64).expect("valid params");
    let client = Box::leak(Box::new(IPIRClient::new(&rlwe, &ypir)));
    let setup = client.generate_setup_simplepir_from_seed(seed_from_u64(args.setup_seed));

    let pt_modulus = ypir.p;
    let db = (0..ypir.db_rows * ypir.db_cols).map(|idx| (idx as u64 % pt_modulus) as u16);
    let server = IPIRServer::<u16>::new(ypir.clone(), db, false, true);
    let offline = server
        .perform_offline_precomputation_simplepir(client.rlwe_params(), &setup.offline_query_polys);
    let preprocessed =
        build_pack_preprocessed_blocks(client.rlwe_params(), &offline.crs_blocks, &setup.key_pair)
            .expect("preprocessing builds");

    let app_data = web::Data::new(ServerState {
        rlwe: client.rlwe_params(),
        server,
        preprocessed,
    });

    println!("Listening on http://127.0.0.1:{}", args.port);
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(app_data.clone())
            .app_data(web::PayloadConfig::new(1usize << 32))
            .service(index)
            .service(query)
            .service(info)
    })
    .workers(1)
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}

#[cfg(not(feature = "http_server"))]
fn main() {
    panic!("This binary requires the 'http_server' feature.");
}

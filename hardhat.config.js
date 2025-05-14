require("@nomicfoundation/hardhat-toolbox");
dotenv = require("dotenv")
dotenv.config()

const accounts = {
  mnemonic: process.env.MNEMONIC || "test test test test test test test test test test test junk",
  accountsBalance: "100000000000000000000000000000",
}

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  networks: {
    hardhat: {
      forking: {
        url: process.env.ETHEREUM_MAINNET_PROVIDER_URL,
        // blockNumber: 20233401,
        // blockNumber: 20825292,
        // blockNumber: 20874859
        // blockNumber: 20892138
        // blockNumber: 20976304
        // blockNumber: 21080765
        // latest
        blockNumber: 21150770
      },
      accounts,
    }
  },
}

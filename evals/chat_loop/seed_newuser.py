"""One fresh user = what onboarding leaves: 1 general wallet + the app's system
categories, zero transactions."""
import sys, uuid, importlib.util, subprocess
spec = importlib.util.spec_from_file_location("seed", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "seed_chat_qa_6m.py"))
seed = importlib.util.module_from_spec(spec); spec.loader.exec_module(seed)
uid = str(uuid.uuid5(seed.NS, "qa-newuser-2026-09-26"))
seed.USER_ID = uid
seed.GENERAL = {"newmain": dict(name="กระเป๋าหลัก", initial=float(sys.argv[1]) if len(sys.argv) > 1 else 5000,
                                 cat="cash", icon=seed.icon_json("assets/cetegory_icons/wallet.png", "0xFFF9E6AE"))}
seed.CARDS, seed.GOALS = {}, {}
seed.ALL_WALLETS = {"newmain": ("general", seed.GENERAL["newmain"])}
cats = seed.build_categories(seed.load_seed_categories())
wallets = seed.build_wallets()
wsid = seed.sid("wallet:newmain")
sql = ["BEGIN;",
       f"INSERT INTO users (id, email) VALUES ('{uid}', 'qa-newuser@chat.test') ON CONFLICT (id) DO NOTHING;",
       f"DELETE FROM transactions WHERE created_by_user_id='{uid}';",
       f"DELETE FROM user_preferences WHERE user_id='{uid}';",
       f"DELETE FROM categories WHERE user_id='{uid}';",
       f"DELETE FROM wallet_members WHERE user_id='{uid}';",
       f"DELETE FROM general_wallets WHERE user_id='{uid}';",
       *wallets, *cats, "COMMIT;"]
url = seed.db_url()
subprocess.run(["psql", url, "-v", "ON_ERROR_STOP=1", "-q"], input="\n".join(sql), text=True, check=True)
print(uid, wsid, len(cats), "categories")

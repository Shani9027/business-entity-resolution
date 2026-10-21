"""
Generates small synthetic train/test datasets that mimic the structure and noise
patterns described in the Business Entity Resolution challenge, purely so the
pipeline can be exercised end-to-end before being run on the real 1GB dataset.
This data is NOT used for the final model / submission in any way.
"""
import random
import pandas as pd

random.seed(42)

US_STREETS = ["Main St", "Oak Ave", "Maple Rd", "1st St", "Broadway", "Park Blvd", "Elm St"]
US_CITIES = [("New York", "NY", "10001"), ("Austin", "TX", "73301"), ("Boston", "MA", "02108"),
             ("Chicago", "IL", "60601"), ("Seattle", "WA", "98101")]
IN_STREETS = ["MG Road", "Nehru Nagar", "Gandhi Chowk", "Station Road", "Ring Road", "Church Street"]
IN_CITIES = [("Mumbai", "Maharashtra", "400001"), ("Pune", "Maharashtra", "411001"),
             ("Bengaluru", "Karnataka", "560001"), ("Delhi", "Delhi", "110001"),
             ("Chennai", "Tamil Nadu", "600001")]
FR_STREETS = ["Rue de Paris", "Avenue Victor Hugo", "Rue Nationale", "Boulevard Saint-Michel"]
FR_CITIES = [("Paris", "Ile-de-France", "75001"), ("Lyon", "Auvergne-Rhone-Alpes", "69001"),
             ("Marseille", "PACA", "13001")]

NAME_STEMS = ["Sharma Traders", "Global Tech Solutions", "Sunrise Bakery", "Metro Auto Works",
              "Bluewater Logistics", "Green Valley Farms", "Prime Electronics", "Star Textiles",
              "National Freight Co", "Silverline Consulting", "Oceanic Imports", "Kumar & Sons",
              "Delta Manufacturing", "Everest Trading", "Coastal Seafoods", "Apex Software Labs"]

LEGAL_SUFFIX_US = ["Inc", "Corp", "Corporation", "LLC", "Co", ""]
LEGAL_SUFFIX_IN = ["Pvt Ltd", "Private Limited", "Ltd", "Limited", ""]
LEGAL_SUFFIX_FR = ["SARL", "SAS", "SA", ""]

ABBREV_MAP_ADDR = {"Road": "Rd", "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd"}


def typo(s, p=0.15):
    if random.random() > p or len(s) < 4:
        return s
    i = random.randint(1, len(s) - 2)
    return s[:i] + s[i+1] + s[i] + s[i+2:]  # swap two adjacent chars


def maybe_abbrev_addr(addr):
    out = addr
    for full, ab in ABBREV_MAP_ADDR.items():
        if full in out and random.random() < 0.5:
            out = out.replace(full, ab)
    return out


def vary_name(name, country):
    n = name
    if random.random() < 0.3:
        n = n.replace("&", "and") if "&" in n else n.replace("and", "&") if " and " in n else n
    if country == "US":
        suf = random.choice(LEGAL_SUFFIX_US)
    elif country == "India":
        suf = random.choice(LEGAL_SUFFIX_IN)
    else:
        suf = random.choice(LEGAL_SUFFIX_FR)
    if suf:
        n = f"{n} {suf}"
    n = typo(n)
    return n


def make_address(country):
    if country == "US":
        num = random.randint(1, 9999)
        street = random.choice(US_STREETS)
        city, state, pin = random.choice(US_CITIES)
        full = f"{num} {street}, {city}, {state} {pin}"
    elif country == "India":
        num = random.randint(1, 999)
        street = random.choice(IN_STREETS)
        city, state, pin = random.choice(IN_CITIES)
        full = f"{num} {street}, {city}, {state} {pin}"
    else:
        num = random.randint(1, 200)
        street = random.choice(FR_STREETS)
        city, region, pin = random.choice(FR_CITIES)
        full = f"{num} {street}, {pin} {city}, {region}"
    return full


def vary_address(addr, drop_prob=0.35):
    parts = [p.strip() for p in addr.split(",")]
    if random.random() < drop_prob and len(parts) > 2:
        # drop a random component (e.g. missing state or pin)
        idx = random.randint(1, len(parts) - 1)
        parts.pop(idx)
    addr2 = ", ".join(parts)
    addr2 = maybe_abbrev_addr(addr2)
    if random.random() < 0.1:
        addr2 = "Near " + random.choice(["City Mall", "Central Park", "Main Station", "SBI ATM"]) + ", " + addr2
    return addr2


def build(n_entities, countries_pool, id_start=1):
    rows_s1, rows_s2, rows_s3, gt = [], [], [], []
    s1_id, s2_id, s3_id = id_start, id_start, id_start
    for _ in range(n_entities):
        country = random.choice(countries_pool)
        base_name = random.choice(NAME_STEMS)
        base_addr = make_address(country)

        this_s1 = f"S1-{s1_id:05d}"; s1_id += 1
        rows_s1.append({"entity_id": this_s1, "business_name": vary_name(base_name, country),
                         "business_address": base_addr, "country": country})

        matched = []
        # decide how many matches in s2/s3 (0,1, or many)
        n_s2 = random.choices([0, 1, 2], weights=[0.35, 0.5, 0.15])[0]
        n_s3 = random.choices([0, 1, 2], weights=[0.35, 0.5, 0.15])[0]
        for _ in range(n_s2):
            eid = f"S2-{s2_id:05d}"; s2_id += 1
            rows_s2.append({"entity_id": eid, "business_name": vary_name(base_name, country),
                             "business_address": vary_address(base_addr), "country": country})
            matched.append(eid)
        for _ in range(n_s3):
            eid = f"S3-{s3_id:05d}"; s3_id += 1
            rows_s3.append({"entity_id": eid, "business_name": vary_name(base_name, country),
                             "business_address": vary_address(base_addr), "country": country})
            matched.append(eid)
        gt.append({"source1_entity_id": this_s1, "matched_entity_ids": ",".join(matched)})

    # add some noise/unrelated S2/S3 records (no match to any S1) to mimic real-world imbalance
    n_noise_s2 = max(3, n_entities // 5)
    n_noise_s3 = max(3, n_entities // 5)
    for _ in range(n_noise_s2):
        country = random.choice(countries_pool)
        eid = f"S2-{s2_id:05d}"; s2_id += 1
        rows_s2.append({"entity_id": eid, "business_name": vary_name(random.choice(NAME_STEMS), country),
                         "business_address": make_address(country), "country": country})
    for _ in range(n_noise_s3):
        country = random.choice(countries_pool)
        eid = f"S3-{s3_id:05d}"; s3_id += 1
        rows_s3.append({"entity_id": eid, "business_name": vary_name(random.choice(NAME_STEMS), country),
                         "business_address": make_address(country), "country": country})

    return rows_s1, rows_s2, rows_s3, gt, s1_id, s2_id, s3_id


if __name__ == "__main__":
    # TRAIN: US + India only (as per problem statement)
    s1, s2, s3, gt, n1, n2, n3 = build(250, ["US", "India"], id_start=1)
    pd.DataFrame(s1).to_csv("dataset/train/train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2).to_csv("dataset/train/train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3).to_csv("dataset/train/train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt).to_csv("dataset/train/train_ground_truth.tsv", sep="\t", index=False)

    # TEST: US + India + France (France unseen in training, per problem statement)
    t1, t2, t3, tgt, _, _, _ = build(120, ["US", "India", "France"], id_start=1)
    pd.DataFrame(t1).to_csv("dataset/test/test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(t2).to_csv("dataset/test/test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(t3).to_csv("dataset/test/test_source3.tsv", sep="\t", index=False)
    # (test ground truth withheld in real challenge; we keep it aside only to self-check the demo)
    pd.DataFrame(tgt).to_csv("dataset/test/_hidden_test_ground_truth_FOR_DEMO_ONLY.tsv", sep="\t", index=False)

    print("Synthetic data generated:")
    print(" train:", len(s1), "S1 /", len(s2), "S2 /", len(s3), "S3")
    print(" test :", len(t1), "S1 /", len(t2), "S2 /", len(t3), "S3")

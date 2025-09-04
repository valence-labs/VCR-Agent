import asyncio

import httpx


class UniProtMapper:
    @classmethod
    async def map(
        cls,
        input_ids: str | list[str],
        from_db: str = "GeneID",
        to_db: str = "UniProtKB",
        tax_id: int = 9606,
    ):
        """
        Map identifiers to UniProt using the UniProt ID Mapping API.

        Args:
            input_ids: single ID or list of IDs (e.g. Entrez or Ensembl).
            from_db: source database name (default "GeneID").
            to_db: target database name (default "UniProtKB").
            tax_id: NCBI taxonomy ID (default 9606 for human).
        """
        if isinstance(input_ids, list):
            ids_str = ",".join(str(i) for i in input_ids)
        else:
            ids_str = str(input_ids)

        async with httpx.AsyncClient() as client:
            # Submit job
            r = await client.post(
                "https://rest.uniprot.org/idmapping/run",
                data={"from": from_db, "to": to_db, "ids": ids_str, "taxId": tax_id},
            )
            r.raise_for_status()
            job_id = r.json().get("jobId")
            if not job_id:
                return None

            # Poll until done
            while True:
                status = (await client.get(f"https://rest.uniprot.org/idmapping/status/{job_id}")).json()
                if status.get("jobStatus") in ("FINISHED", "COMPLETE") or "results" in status:
                    break
                await asyncio.sleep(1)

            # Get results
            res = await client.get(f"https://rest.uniprot.org/idmapping/results/{job_id}")
            res.raise_for_status()
            results = res.json().get("results", [])
            if not results:
                return None

            # Create instance and populate fields
            mapper = dict()
            mapper["uniprot_id"] = results[0].get("to")

            # Hydrate with entry data
            entry = (await client.get(f"https://rest.uniprot.org/uniprotkb/{mapper['uniprot_id']}")).json()
            mapper["uniprot_name"] = entry.get("uniProtkbId")
            mapper["uniprot_function_summary"] = ".".join(
                [
                    t["value"].strip()
                    for c in (entry.get("comments") or [])
                    if c.get("commentType") == "FUNCTION" and not c.get("molecule")
                    for t in (c.get("texts") or [])
                    if t.get("value")
                ]
            )

            for x in entry.get("uniProtKBCrossReferences", []):
                db = x.get("database")
                if db == "ChEMBL":
                    mapper["chembl_id"] = x.get("id")
                elif db == "HPA":
                    mapper["hpa_id"] = x.get("id")

        return mapper

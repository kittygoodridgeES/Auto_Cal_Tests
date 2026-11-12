import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv


class AirtableReader:
    """Fetch a single field value from an Airtable record.

    Tables:
    All job data: "jobs"
    Diamond tracking: "Location Update Events"
    Curring head tracking: "Heads in Production"
    """

    BASE_URL = "https://api.airtable.com/v0"

    def __init__(self, tables: dict[str, str], timeout: float = 10.0):
        """
        tables: maps each table name (or tbl... ID) to the base ID (app...) it lives in,
                e.g. {"Jobs": "appAAAAAAAAAAAAAA", "Machines": "appBBBBBBBBBBBBBB"}
        """
        load_dotenv()
        self.tables = tables
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {os.environ["AIRTABLE_ACCESS_TOKEN"]}"})

    def diamond_length_change(self, slot_id):
        """
        Takes in a slot and calculates the difference between the most diamond in the cutter and the most recent diamond
        Change in diamond length = previous diamond length - current diamond length 
        """
        diamond_id_new = self.get_diamond_ID_from_location(slot_id)
        if diamond_id_new == None:
            #there is no current diamond logged in the cutter
            return None
        diamond_id_previous = self.get_previous_diamond_ID(slot_id)
        if diamond_id_previous == None:
            #there has only ever been one diamond in the cutter
            return False
        diamond_new_record = self.find_by_field("Length Update Events", "Diamond Records 2", diamond_id_new, view=os.environ["DIAMOND_LENGTH_VIEW_ID"], newest_by="Date")
        diamond_previous_record = self.find_by_field("Length Update Events", "Diamond Records 2", diamond_id_previous, view=os.environ["DIAMOND_LENGTH_VIEW_ID"], newest_by="Date")

        diamond_length_new = diamond_new_record["fields"]["latestLength"]
        diamond_length_previous = diamond_previous_record["fields"]["latestLength"]
        change = float(diamond_length_previous) - float(diamond_length_new)
        change = round(change,3)
        
        return change

    def get_previous_diamond_ID(self, slot_id):
        """
        Takes in slot ID and returns the second most recent diamond ID
        """
        if slot_id.startswith("1."):
            slot_id = slot_id[2:]
        
        location = "slot_" + slot_id
        try:
            record = self.find_by_field("Location Update Events", "New Location", location, view=os.environ["DIAMOND_LOACTION_VIEW_ID"], newest_by="Date", nth=1)
            
        except IndexError:
            return None
        if record is None:
            return None
        
        second_diamond = record["fields"]["Diamond ID (from Diamond Records 4)"][0]
        return second_diamond
    

    def get_diamond_ID_from_location(self, slot_id):
        """ 
        takes in a slot id in the form "1.2", "2.1.1" etc.. and returns diamond ID 
        """
        if slot_id.startswith("1."):
            slot_id = slot_id[2:]
        location = "slot_" + slot_id
        try:
            record = self.find_by_field("Location Update Events", "New Location", location, view=os.environ["DIAMOND_LOACTION_VIEW_ID"], newest_by="Date")
        except IndexError:
            return None
        if record is None:
            return None

        diamond_id = record["fields"]["Diamond ID (from Diamond Records 4)"][0]
        return diamond_id


    def _base_for(self, table: str) -> str:
        try:
            return self.tables[table]
        except KeyError:
            raise KeyError(
                f"No base ID configured for table '{table}'. "
                f"Known tables: {', '.join(self.tables)}"
            ) from None

    def get_record(self, table: str, record_id: str) -> dict:
        """Return the full record dict: {'id', 'createdTime', 'fields'}."""
        base_id = self._base_for(table)
        url = f"{self.BASE_URL}/{base_id}/{quote(table, safe='')}/{record_id}"
        resp = self.session.get(url, timeout=self.timeout)

        if resp.status_code == 404:
            raise KeyError(f"Record {record_id} not found in {base_id}/{table}")
        resp.raise_for_status()
        return resp.json()

    def get_field(self, table: str, record_id: str, field: str, default=None):
        """
        Return one field's value from a record.

        Airtable omits empty fields ("", [], false) from the response,
        so a missing key returns `default` rather than raising.
        """
        return self.get_record(table, record_id)["fields"].get(field, default)

    def find_by_field(self, table: str, field: str, value, view: str | None = None,
                    newest_by: str | None = None, nth: int = 0) -> dict | None:
        """
        Return a record where `field` equals `value`, optionally within a view.
        If `newest_by` is given (e.g. "Date"), records are sorted newest-first and
        `nth` picks which one: 0 = most recent, 1 = second most recent, etc.
        """
        base_id = self._base_for(table)
        url = f"{self.BASE_URL}/{base_id}/{quote(table, safe='')}"

        if isinstance(value, (int, float)):
            formula = f"{{{field}}} = {value}"
        else:
            formula = f"{{{field}}} = '{value}'"

        params = {"filterByFormula": formula, "maxRecords": nth + 1}
        if view:
            params["view"] = view
        if newest_by:
            params["sort[0][field]"] = newest_by
            params["sort[0][direction]"] = "desc"

        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        records = resp.json()["records"]
        return records[nth] if len(records) > nth else None


# if __name__ == "__main__":
#     load_dotenv()
#     reader = AirtableReader(
#         tables={
#             "Jobs": os.environ["JOB_TABLE_ID"],
#             "Location Update Events": os.environ["AT_DIAMOND_BASE_ID"],
#             "Length Update Events": os.environ["AT_DIAMOND_BASE_ID"],
#         },
#     )
#     #print(reader.get_diamond_ID_from_location("3.1.1"))
#     print(reader.diamond_length_change("3.2.2"))

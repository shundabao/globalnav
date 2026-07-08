"""Deterministic sampler for GlobNav-Bench pilot data.

The sampler creates globally distributed route skeletons first, then attaches
natural-language instructions and labels for the three main benchmark uses:
planning feasibility, intent/clarification, and hybrid follower evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from globe_nav.bench.schema import SCHEMA_VERSION, validate_example
from globe_nav.maps.geocoder import haversine_km
from globe_nav.planner.flights import FlightGraph


@dataclass(frozen=True)
class POI:
    label: str
    query: str
    lat: float
    lon: float
    poi_type: str
    osm_tags: tuple[str, ...]


@dataclass(frozen=True)
class CitySeed:
    city_id: str
    city: str
    country: str
    macro_region: str
    language_hint: str
    lat: float
    lon: float
    airport_iata: str
    gtfs_coverage: str
    pois: tuple[POI, ...]


def _poi(label: str, query: str, lat: float, lon: float, poi_type: str, *tags: str) -> POI:
    return POI(label, query, lat, lon, poi_type, tags)


CITY_SEEDS: tuple[CitySeed, ...] = (
    CitySeed('sydney', 'Sydney', 'Australia', 'Oceania', 'en', -33.8688, 151.2093, 'SYD', 'available', (
        _poi('AUS_CITY_CENTER_HOTEL', 'Hyatt Regency Sydney, Sydney, Australia', -33.8699, 151.2026, 'hotel', 'tourism=hotel'),
        _poi('AUS_WATERFRONT_LANDMARK', 'Sydney Opera House, Sydney, Australia', -33.8568, 151.2153, 'landmark', 'tourism=attraction'),
        _poi('AUS_MAIN_RAIL_STATION', 'Central Station, Sydney, Australia', -33.8830, 151.2067, 'station', 'railway=station'),
        _poi('AUS_FERRY_TERMINAL', 'Circular Quay ferry wharf, Sydney, Australia', -33.8610, 151.2111, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('new_york', 'New York', 'United States', 'North America', 'en', 40.7128, -74.0060, 'JFK', 'available', (
        _poi('NYC_CONFERENCE_VENUE', 'Jacob K. Javits Convention Center, New York, USA', 40.7570, -74.0024, 'venue', 'amenity=events_venue'),
        _poi('NYC_TIMES_SQUARE', 'Times Square, New York, USA', 40.7580, -73.9855, 'landmark', 'tourism=attraction'),
        _poi('NYC_RAIL_STATION', 'Penn Station, New York, USA', 40.7506, -73.9935, 'station', 'railway=station'),
        _poi('NYC_FERRY_PIER', 'Whitehall Terminal, New York, USA', 40.7010, -74.0132, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('london', 'London', 'United Kingdom', 'Europe', 'en', 51.5072, -0.1276, 'LHR', 'available', (
        _poi('LON_CITY_HOTEL', 'The Royal Horseguards Hotel, London, UK', 51.5062, -0.1234, 'hotel', 'tourism=hotel'),
        _poi('LON_CONFERENCE_CENTER', 'ExCeL London, London, UK', 51.5081, 0.0297, 'venue', 'amenity=events_venue'),
        _poi('LON_RAIL_STATION', 'London St Pancras International, London, UK', 51.5314, -0.1261, 'station', 'railway=station'),
        _poi('LON_RIVER_PIER', 'Westminster Pier, London, UK', 51.5012, -0.1238, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('paris', 'Paris', 'France', 'Europe', 'fr', 48.8566, 2.3522, 'CDG', 'available', (
        _poi('PAR_CITY_HOTEL', 'Hotel de Ville, Paris, France', 48.8566, 2.3522, 'hotel', 'tourism=hotel'),
        _poi('PAR_EXHIBITION_CENTER', 'Paris Expo Porte de Versailles, Paris, France', 48.8303, 2.2873, 'venue', 'amenity=events_venue'),
        _poi('PAR_RAIL_STATION', 'Gare du Nord, Paris, France', 48.8809, 2.3553, 'station', 'railway=station'),
        _poi('PAR_LANDMARK', 'Louvre Museum, Paris, France', 48.8606, 2.3376, 'landmark', 'tourism=museum'),
    )),
    CitySeed('tokyo', 'Tokyo', 'Japan', 'Asia', 'ja', 35.6762, 139.6503, 'HND', 'available', (
        _poi('TYO_CITY_HOTEL', 'Tokyo Station Hotel, Tokyo, Japan', 35.6810, 139.7667, 'hotel', 'tourism=hotel'),
        _poi('TYO_CONVENTION_CENTER', 'Tokyo International Forum, Tokyo, Japan', 35.6768, 139.7648, 'venue', 'amenity=events_venue'),
        _poi('TYO_RAIL_STATION', 'Tokyo Station, Tokyo, Japan', 35.6812, 139.7671, 'station', 'railway=station'),
        _poi('TYO_FERRY_TERMINAL', 'Takeshiba Pier, Tokyo, Japan', 35.6546, 139.7621, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('singapore', 'Singapore', 'Singapore', 'Asia', 'en', 1.3521, 103.8198, 'SIN', 'available', (
        _poi('SG_CITY_HOTEL', 'Raffles Hotel Singapore', 1.2949, 103.8547, 'hotel', 'tourism=hotel'),
        _poi('SG_CONVENTION_CENTER', 'Sands Expo and Convention Centre, Singapore', 1.2839, 103.8607, 'venue', 'amenity=events_venue'),
        _poi('SG_RAIL_STATION', 'City Hall MRT Station, Singapore', 1.2932, 103.8520, 'station', 'railway=station'),
        _poi('SG_FERRY_TERMINAL', 'HarbourFront Ferry Terminal, Singapore', 1.2643, 103.8207, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('dubai', 'Dubai', 'United Arab Emirates', 'Middle East', 'en', 25.2048, 55.2708, 'DXB', 'available', (
        _poi('DXB_CITY_HOTEL', 'Dubai Creek hotel district, Dubai, UAE', 25.2595, 55.3088, 'hotel', 'tourism=hotel'),
        _poi('DXB_CONVENTION_CENTER', 'Dubai World Trade Centre, Dubai, UAE', 25.2285, 55.2885, 'venue', 'amenity=events_venue'),
        _poi('DXB_METRO_STATION', 'Burj Khalifa Dubai Mall Metro Station, Dubai, UAE', 25.2012, 55.2696, 'station', 'railway=station'),
        _poi('DXB_FERRY_TERMINAL', 'Dubai Marina ferry station, Dubai, UAE', 25.0772, 55.1355, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('san_francisco', 'San Francisco', 'United States', 'North America', 'en', 37.7749, -122.4194, 'SFO', 'available', (
        _poi('SFO_WATERFRONT_HOTEL', 'Hyatt Regency San Francisco, San Francisco, USA', 37.7942, -122.3957, 'hotel', 'tourism=hotel'),
        _poi('SFO_CONFERENCE_VENUE', 'Moscone Center, San Francisco, USA', 37.7840, -122.4011, 'venue', 'amenity=events_venue'),
        _poi('SFO_RAIL_STATION', 'San Francisco Caltrain Station, San Francisco, USA', 37.7766, -122.3947, 'station', 'railway=station'),
        _poi('SFO_FERRY_TERMINAL', 'San Francisco Ferry Building, San Francisco, USA', 37.7955, -122.3937, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
    CitySeed('cape_town', 'Cape Town', 'South Africa', 'Africa', 'en', -33.9249, 18.4241, 'CPT', 'limited', (
        _poi('CPT_CITY_HOTEL', 'Cape Town city centre hotel district, Cape Town, South Africa', -33.9198, 18.4217, 'hotel', 'tourism=hotel'),
        _poi('CPT_CONVENTION_CENTER', 'Cape Town International Convention Centre, Cape Town, South Africa', -33.9154, 18.4258, 'venue', 'amenity=events_venue'),
        _poi('CPT_RAIL_STATION', 'Cape Town railway station, Cape Town, South Africa', -33.9227, 18.4256, 'station', 'railway=station'),
        _poi('CPT_WATERFRONT', 'V&A Waterfront, Cape Town, South Africa', -33.9068, 18.4207, 'landmark', 'tourism=attraction'),
    )),
    CitySeed('nairobi', 'Nairobi', 'Kenya', 'Africa', 'en', -1.2921, 36.8219, 'NBO', 'limited', (
        _poi('NBO_CITY_HOTEL', 'Nairobi city centre hotel district, Nairobi, Kenya', -1.2864, 36.8172, 'hotel', 'tourism=hotel'),
        _poi('NBO_CONFERENCE_CENTER', 'Kenyatta International Convention Centre, Nairobi, Kenya', -1.2881, 36.8231, 'venue', 'amenity=events_venue'),
        _poi('NBO_RAIL_STATION', 'Nairobi railway station, Nairobi, Kenya', -1.2922, 36.8296, 'station', 'railway=station'),
        _poi('NBO_PARK_GATE', 'Nairobi National Park main gate, Nairobi, Kenya', -1.3510, 36.8300, 'landmark', 'tourism=attraction'),
    )),
    CitySeed('sao_paulo', 'Sao Paulo', 'Brazil', 'South America', 'pt', -23.5558, -46.6396, 'GRU', 'available', (
        _poi('SAO_CITY_HOTEL', 'Avenida Paulista hotel district, Sao Paulo, Brazil', -23.5617, -46.6562, 'hotel', 'tourism=hotel'),
        _poi('SAO_EXPO_CENTER', 'Sao Paulo Expo, Sao Paulo, Brazil', -23.6464, -46.6308, 'venue', 'amenity=events_venue'),
        _poi('SAO_RAIL_STATION', 'Luz Station, Sao Paulo, Brazil', -23.5352, -46.6359, 'station', 'railway=station'),
        _poi('SAO_LANDMARK', 'Ibirapuera Park, Sao Paulo, Brazil', -23.5874, -46.6576, 'landmark', 'tourism=attraction'),
    )),
    CitySeed('istanbul', 'Istanbul', 'Turkey', 'Middle East', 'tr', 41.0082, 28.9784, 'IST', 'available', (
        _poi('IST_CITY_HOTEL', 'Sultanahmet hotel district, Istanbul, Turkey', 41.0054, 28.9768, 'hotel', 'tourism=hotel'),
        _poi('IST_CONVENTION_CENTER', 'Istanbul Congress Center, Istanbul, Turkey', 41.0483, 28.9882, 'venue', 'amenity=events_venue'),
        _poi('IST_RAIL_STATION', 'Sirkeci railway station, Istanbul, Turkey', 41.0148, 28.9769, 'station', 'railway=station'),
        _poi('IST_FERRY_TERMINAL', 'Eminonu ferry terminal, Istanbul, Turkey', 41.0178, 28.9708, 'ferry_terminal', 'amenity=ferry_terminal'),
    )),
)


AMBIGUOUS_AIRPORTS: dict[str, list[str]] = {
    'London': ['Heathrow', 'Gatwick', 'Stansted', 'Luton', 'London City'],
    'New York': ['JFK', 'LaGuardia', 'Newark'],
    'Tokyo': ['Haneda', 'Narita'],
    'Paris': ['Charles de Gaulle', 'Orly', 'Beauvais'],
}


class GlobNavBenchSampler:
    """Create deterministic GlobNav-Bench examples from global seed sources."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.flights = FlightGraph()
        self.flights.ensure_loaded()
        self._cities = [c for c in CITY_SEEDS if c.airport_iata in self.flights.airports]

    def sample(self, count: int = 500) -> list[dict[str, Any]]:
        quotas = self._quotas(count)
        examples: list[dict[str, Any]] = []
        examples.extend(self._planning_examples(quotas['planning_feasibility']))
        examples.extend(self._intent_examples(quotas['intent_clarification']))
        examples.extend(self._route_option_examples(quotas['route_option_generation']))
        examples.extend(self._follower_examples(quotas['hybrid_follower']))
        examples.extend(self._stress_examples(quotas['stress_test']))
        for i, ex in enumerate(examples[:count], 1):
            ex['id'] = f'gnb_{i:05d}_{ex["split"]}'
            errors = validate_example(ex)
            if errors:
                raise ValueError(f'invalid generated example {ex["id"]}: {errors}')
        return examples[:count]

    @staticmethod
    def _quotas(count: int) -> dict[str, int]:
        ratios = {
            'planning_feasibility': 0.30,
            'intent_clarification': 0.40,
            'route_option_generation': 0.10,
            'hybrid_follower': 0.15,
            'stress_test': 0.05,
        }
        quotas = {k: int(count * v) for k, v in ratios.items()}
        while sum(quotas.values()) < count:
            quotas['intent_clarification'] += 1
        return quotas

    def _planning_examples(self, n: int) -> list[dict[str, Any]]:
        examples = []
        templates = [
            'I am at {origin}, facing the main avenue. Get me to {dest}; use a flight if needed.',
            'Plan a global route from {origin} to {dest}, including airport access and final ground transfer.',
            'Starting at {origin}, I need to reach {dest}. Show the feasible multimodal choices.',
            'From {origin}, take me to {dest}; avoid hallucinated direct flights.',
        ]
        for idx in range(n):
            origin_city, dest_city = self._city_pair(different_region=idx % 2 == 0)
            origin = self._pick_poi(origin_city, ('hotel', 'landmark', 'venue'))
            dest = self._pick_poi(dest_city, ('venue', 'landmark', 'hotel'))
            path = self._flight_path(origin_city, dest_city)
            ex = self._base(
                split='planning_feasibility',
                language='en',
                instruction=self.rng.choice(templates).format(
                    origin=origin.label.lower().replace('_', ' '),
                    dest=dest.label.lower().replace('_', ' '),
                ),
                categories=['global_hybrid', 'llm_only_vs_environment'],
                origin=origin,
                destination=dest,
                source_cities=(origin_city, dest_city),
            )
            ex['route_annotation'] = self._route_annotation(origin_city, dest_city, origin, dest, path)
            ex['metadata']['llm_only_pitfalls'] = [
                'invented direct flight',
                'missing airport access',
                'missing egress transfer',
                'ungrounded duration estimate',
            ]
            examples.append(ex)
        return examples

    def _intent_examples(self, n: int) -> list[dict[str, Any]]:
        examples = []
        for idx in range(n):
            if idx % 4 == 0:
                examples.append(self._ambiguous_airport_example())
                continue
            if idx % 4 == 1:
                examples.append(self._missing_endpoint_example(missing_origin=True))
                continue
            if idx % 4 == 2:
                examples.append(self._missing_endpoint_example(missing_origin=False))
                continue
            origin_city, dest_city = self._city_pair()
            origin = self._pick_poi(origin_city, ('hotel', 'station'))
            dest = self._pick_poi(dest_city, ('venue', 'landmark'))
            language = self.rng.choice(['en', 'zh', 'mixed'])
            instruction = self._intent_instruction(language, origin, dest)
            ex = self._base(
                split='intent_clarification',
                language=language,
                instruction=instruction,
                categories=['intent_parsing', 'bilingual_complex'],
                origin=origin,
                destination=dest,
                source_cities=(origin_city, dest_city),
            )
            ex['clarification'] = {
                'needs_clarification': False,
                'gold_question_type': None,
                'gold_question': None,
                'accepted_origin_strings': [origin.label, origin.query],
                'accepted_destination_strings': [dest.label, dest.query],
            }
            ex['route_annotation'] = self._route_annotation(origin_city, dest_city, origin, dest, self._flight_path(origin_city, dest_city))
            examples.append(ex)
        return examples

    def _route_option_examples(self, n: int) -> list[dict[str, Any]]:
        examples = []
        templates = [
            'Show me all reasonable ways from {origin} to {dest}.',
            'I want route choices from {origin} to {dest}, not just the fastest path.',
            'Compare drive, walk, and public transport from {origin} to {dest}.',
        ]
        for _ in range(n):
            city = self.rng.choice(self._cities)
            origin = self._pick_poi(city, ('hotel', 'station'))
            dest = self._pick_poi(city, ('venue', 'landmark', 'ferry_terminal'))
            if origin == dest:
                dest = self._pick_poi(city, ('venue', 'landmark'))
            ex = self._base(
                split='route_option_generation',
                language='en',
                instruction=self.rng.choice(templates).format(
                    origin=origin.label.lower().replace('_', ' '),
                    dest=dest.label.lower().replace('_', ' '),
                ),
                categories=['segment_options', 'local_multimodal'],
                origin=origin,
                destination=dest,
                source_cities=(city, city),
            )
            ex['route_annotation'] = {
                'expected_segment_order': ['seg_local'],
                'environment_requirements': [
                    'return one drive option when OSRM driving is available',
                    'return top five non-drive options when enough graph paths exist',
                    'always retain pure walking',
                    'deduplicate by mode chain and transfer stop sequence',
                ],
                'segments': [{
                    'segment_id': 'seg_local',
                    'from': origin.label,
                    'to': dest.label,
                    'allowed_modes': self._local_modes(city),
                    'expected_options': {
                        'drive': True,
                        'pure_walk_required': True,
                        'non_drive_top_k': 5,
                    },
                    'evidence': ['OSRM', 'OSM/Overpass', self._gtfs_label(city)],
                }],
            }
            examples.append(ex)
        return examples

    def _follower_examples(self, n: int) -> list[dict[str, Any]]:
        examples = []
        templates = [
            'Starting at {origin}, drive to the international airport. After the flight, transfer to the ferry terminal, take the ferry, then walk to {dest}.',
            'Begin outside {origin}. Follow the co-driver notes to the airport, fly overseas, ride the ferry phase, and finish on foot at {dest}.',
            'From {origin}, complete a hybrid route: drive, flight, ferry, then walk to {dest}.',
        ]
        for idx in range(n):
            origin_city, dest_city = self._city_pair(different_region=True)
            origin = self._pick_poi(origin_city, ('hotel',))
            dest = self._pick_poi(dest_city, ('venue', 'landmark'))
            path = self._flight_path(origin_city, dest_city)
            ex = self._base(
                split='hybrid_follower',
                language='en',
                instruction=self.rng.choice(templates).format(
                    origin=origin.label.lower().replace('_', ' '),
                    dest=dest.label.lower().replace('_', ' '),
                ),
                categories=['hybrid_follower', 'drive_flight_ferry_walk', 'llm_decision_agent'],
                origin=origin,
                destination=dest,
                source_cities=(origin_city, dest_city),
            )
            ex['route_annotation'] = self._route_annotation(
                origin_city, dest_city, origin, dest, path,
                include_ferry=idx % 2 == 0,
                follower=True,
            )
            ex['follower_annotation'] = self._follower_annotation(origin_city, dest_city, include_ferry=idx % 2 == 0)
            examples.append(ex)
        return examples

    def _stress_examples(self, n: int) -> list[dict[str, Any]]:
        examples = []
        for idx in range(n):
            if idx % 2 == 0:
                ex = self._ambiguous_place_example()
            else:
                ex = self._transport_preference_example()
            ex['split'] = 'stress_test'
            ex['categories'] = list(set(ex['categories'] + ['hard_negative']))
            examples.append(ex)
        return examples

    def _base(
        self,
        split: str,
        language: str,
        instruction: str,
        categories: list[str],
        origin: POI | None,
        destination: POI | None,
        source_cities: tuple[CitySeed | None, CitySeed | None],
    ) -> dict[str, Any]:
        origin_city, dest_city = source_cities
        return {
            'schema_version': SCHEMA_VERSION,
            'id': 'pending',
            'split': split,
            'language': language,
            'instruction': instruction,
            'categories': categories,
            'gold_intent': {
                'origin': self._place(origin) if origin else None,
                'destination': self._place(destination) if destination else None,
            },
            'clarification': {
                'needs_clarification': False,
                'gold_question_type': None,
                'gold_question': None,
            },
            'route_annotation': {'expected_segment_order': [], 'environment_requirements': [], 'segments': []},
            'follower_annotation': None,
            'metadata': {
                'origin_city': origin_city.city if origin_city else None,
                'destination_city': dest_city.city if dest_city else None,
                'macro_regions': [
                    origin_city.macro_region if origin_city else None,
                    dest_city.macro_region if dest_city else None,
                ],
                'source_stack': ['GeoNames-style city seeds', 'OSM/Overpass POI tags', 'OpenFlights', 'GTFS metadata'],
                'review_status': 'machine_validated_seed_requires_human_review',
            },
            'annotation_status': 'pilot_seed',
        }

    @staticmethod
    def _place(poi: POI | None) -> dict[str, Any] | None:
        if poi is None:
            return None
        return {
            'anonymized_label': poi.label,
            'canonical_query': poi.query,
            'lat': poi.lat,
            'lon': poi.lon,
            'place_type': poi.poi_type,
            'osm_tags': list(poi.osm_tags),
        }

    def _route_annotation(
        self,
        origin_city: CitySeed,
        dest_city: CitySeed,
        origin: POI,
        dest: POI,
        flight_path: list[str],
        include_ferry: bool = False,
        follower: bool = False,
    ) -> dict[str, Any]:
        order = ['seg_access', 'seg_flight', 'seg_egress']
        if include_ferry:
            order = ['seg_access_drive', 'seg_flight', 'seg_transfer_drive', 'seg_ferry', 'seg_final_walk']
        segments = []
        if include_ferry:
            segments.extend([
                self._segment('seg_access_drive', origin.label, f'{origin_city.airport_iata}_AIRPORT', ['drive'], ['OSRM driving route']),
                self._segment('seg_flight', origin_city.airport_iata, dest_city.airport_iata, ['fly'], [self._flight_evidence(flight_path)]),
                self._segment('seg_transfer_drive', f'{dest_city.airport_iata}_AIRPORT', f'{dest_city.city.upper()}_FERRY_TERMINAL', ['drive'], ['OSRM driving route']),
                self._segment('seg_ferry', f'{dest_city.city.upper()}_FERRY_TERMINAL', f'{dest_city.city.upper()}_DOWNTOWN_PIER', ['ferry'], ['OSM amenity=ferry_terminal', 'estimated phase duration']),
                self._segment('seg_final_walk', f'{dest_city.city.upper()}_DOWNTOWN_PIER', dest.label, ['walk'], ['OSRM walking route']),
            ])
        else:
            segments.extend([
                self._segment('seg_access', origin.label, f'{origin_city.airport_iata}_AIRPORT', self._local_modes(origin_city), ['OSRM', 'OSM/Overpass', self._gtfs_label(origin_city)]),
                self._segment('seg_flight', origin_city.airport_iata, dest_city.airport_iata, ['fly'], [self._flight_evidence(flight_path)]),
                self._segment('seg_egress', f'{dest_city.airport_iata}_AIRPORT', dest.label, self._local_modes(dest_city), ['OSRM', 'OSM/Overpass', self._gtfs_label(dest_city)]),
            ])
        return {
            'expected_segment_order': order,
            'environment_requirements': [
                'geocode endpoints',
                'verify OpenFlights path or mark estimated fallback',
                'enumerate one drive option plus top five non-drive local options',
                'retain pure walking for local segments',
            ],
            'segments': segments,
            'flight_path': flight_path,
            'direct_flight_verified': len(flight_path) == 2 and self.flights.has_direct(flight_path[0], flight_path[1]),
            'follower_ready': follower,
        }

    @staticmethod
    def _segment(segment_id: str, src: str, dst: str, modes: list[str], evidence: list[str]) -> dict[str, Any]:
        return {
            'segment_id': segment_id,
            'from': src,
            'to': dst,
            'allowed_modes': modes,
            'default_mode': modes[0] if modes else None,
            'evidence': evidence,
            'annotation_status': 'seed_verified_by_rules',
        }

    def _follower_annotation(self, origin_city: CitySeed, dest_city: CitySeed, include_ferry: bool) -> dict[str, Any]:
        oracle = ['forward', 'right', 'forward', 'board', 'takeoff', 'cruise', 'land']
        transitions = [{
            'segment_id': 'seg_flight',
            'oracle_actions': ['board', 'takeoff', 'cruise', 'land'],
            'success_condition': f'arrive at {dest_city.airport_iata}',
        }]
        if include_ferry:
            oracle.extend(['forward', 'depart', 'cruise', 'dock'])
            transitions.append({
                'segment_id': 'seg_ferry',
                'oracle_actions': ['depart', 'cruise', 'dock'],
                'success_condition': 'downtown ferry pier reached',
            })
        oracle.extend(['forward', 'left', 'forward'])
        return {
            'evaluation_target': 'LLM decision agent, not oracle simulator replay',
            'system_simulation_oracle': oracle,
            'phase_transitions': transitions,
            'procedural_notes': [
                {
                    'segment_id': 'seg_access_drive',
                    'notes': [
                        f'Leave the origin and follow airport signs toward {origin_city.airport_iata}.',
                        'At the third major intersection, keep right for the airport access road.',
                        'Stop at the departures curb before boarding.',
                    ],
                },
                {
                    'segment_id': 'seg_transfer_drive',
                    'notes': [
                        'Exit the arrivals area and follow signs for the waterfront or city center transfer.',
                        'Turn right at the first major intersection after the terminal exit.',
                        'Continue until the ferry or downtown transfer point is visible ahead.',
                    ],
                },
                {
                    'segment_id': 'seg_final_walk',
                    'notes': [
                        'Walk inland from the arrival point along the main pedestrian corridor.',
                        'Turn left when the destination facade is visible.',
                        'Stop at the main entrance.',
                    ],
                },
            ],
            'metrics': ['success_rate', 'segment_success', 'action_accuracy', 'mode_transition_accuracy', 'step_efficiency'],
            'streetview_required': False,
        }

    def _ambiguous_airport_example(self) -> dict[str, Any]:
        city_name = self.rng.choice(list(AMBIGUOUS_AIRPORTS))
        city = next((c for c in self._cities if c.city == city_name), self._cities[0])
        origin = self._pick_poi(city, ('hotel', 'station'))
        ex = self._base(
            split='intent_clarification',
            language='en',
            instruction=f'Take me to the airport from {city_name}.',
            categories=['ambiguous_instruction', 'clarification'],
            origin=origin,
            destination=None,
            source_cities=(city, None),
        )
        airports = ', '.join(AMBIGUOUS_AIRPORTS[city_name])
        ex['clarification'] = {
            'needs_clarification': True,
            'gold_question_type': 'ambiguous_airport',
            'gold_question': f'Which {city_name}-area airport do you want to go to, for example {airports}?',
            'acceptable_questions': [
                f'Which airport in {city_name} do you mean?',
                f'Do you mean one of these airports: {airports}?',
            ],
        }
        return ex

    def _missing_endpoint_example(self, missing_origin: bool) -> dict[str, Any]:
        city_a, city_b = self._city_pair()
        origin = None if missing_origin else self._pick_poi(city_a, ('hotel', 'station'))
        dest = self._pick_poi(city_b, ('venue', 'landmark')) if missing_origin else None
        instruction = (
            f'I need to get to {dest.label.lower().replace("_", " ")} as soon as possible.'
            if missing_origin else
            f'I am at {origin.label.lower().replace("_", " ")} and need a global route.'
        )
        ex = self._base(
            split='intent_clarification',
            language='en',
            instruction=instruction,
            categories=['ambiguous_instruction', 'clarification'],
            origin=origin,
            destination=dest,
            source_cities=(city_a if origin else None, city_b if dest else None),
        )
        ex['clarification'] = {
            'needs_clarification': True,
            'gold_question_type': 'missing_origin' if missing_origin else 'missing_destination',
            'gold_question': 'Where are you starting from?' if missing_origin else 'Where do you want to go?',
            'acceptable_questions': ['What is your starting point?'] if missing_origin else ['What is your destination?'],
        }
        return ex

    def _ambiguous_place_example(self) -> dict[str, Any]:
        city = self.rng.choice(self._cities)
        origin = self._pick_poi(city, ('hotel',))
        ex = self._base(
            split='intent_clarification',
            language='en',
            instruction='I am downtown. Take me to Central Station.',
            categories=['ambiguous_place', 'clarification'],
            origin=origin,
            destination=None,
            source_cities=(city, None),
        )
        ex['clarification'] = {
            'needs_clarification': True,
            'gold_question_type': 'ambiguous_place',
            'gold_question': 'Which city or exact Central Station do you mean?',
            'acceptable_questions': ['Which Central Station do you mean?', 'Can you provide the city for Central Station?'],
        }
        return ex

    def _transport_preference_example(self) -> dict[str, Any]:
        city_a, city_b = self._city_pair(different_region=True)
        origin = self._pick_poi(city_a, ('hotel',))
        dest = self._pick_poi(city_b, ('venue',))
        ex = self._base(
            split='intent_clarification',
            language='en',
            instruction=f'I am at {origin.label.lower().replace("_", " ")} and need to reach {dest.label.lower().replace("_", " ")} today, but I might need to avoid flying.',
            categories=['transport_preference', 'clarification'],
            origin=origin,
            destination=dest,
            source_cities=(city_a, city_b),
        )
        ex['clarification'] = {
            'needs_clarification': True,
            'gold_question_type': 'transport_preference',
            'gold_question': 'Should I avoid flights even if that makes the trip much longer?',
            'acceptable_questions': ['Do you want to avoid flying?', 'Is flying allowed for this trip?'],
        }
        return ex

    def _city_pair(self, different_region: bool = False) -> tuple[CitySeed, CitySeed]:
        origin = self.rng.choice(self._cities)
        candidates = [c for c in self._cities if c.city_id != origin.city_id]
        if different_region:
            regional = [c for c in candidates if c.macro_region != origin.macro_region]
            if regional:
                candidates = regional
        dest = self.rng.choice(candidates)
        return origin, dest

    def _pick_poi(self, city: CitySeed, types: tuple[str, ...]) -> POI:
        candidates = [p for p in city.pois if p.poi_type in types]
        return self.rng.choice(candidates or list(city.pois))

    def _flight_path(self, origin_city: CitySeed, dest_city: CitySeed) -> list[str]:
        paths = self.flights.find_paths(origin_city.airport_iata, dest_city.airport_iata, max_stops=1)
        if paths:
            return paths[0]
        estimated = self.flights.estimate_via_hubs(origin_city.airport_iata, dest_city.airport_iata)
        return estimated[0] if estimated else [origin_city.airport_iata, dest_city.airport_iata]

    def _flight_evidence(self, path: list[str]) -> str:
        if len(path) == 2 and self.flights.has_direct(path[0], path[1]):
            return f'OpenFlights direct route: {path[0]}-{path[1]}'
        if len(path) > 2:
            verified_edges = all(self.flights.has_direct(path[i], path[i + 1]) for i in range(len(path) - 1))
            if verified_edges:
                return 'OpenFlights multi-leg route: ' + '->'.join(path)
        return 'estimated hub route: ' + '->'.join(path)

    @staticmethod
    def _local_modes(city: CitySeed) -> list[str]:
        modes = ['drive', 'walk', 'bus', 'walk+bus', 'walk+train']
        if city.gtfs_coverage == 'available':
            modes.append('walk+tram')
        return modes

    @staticmethod
    def _gtfs_label(city: CitySeed) -> str:
        return 'Mobility Database/GTFS coverage available' if city.gtfs_coverage == 'available' else 'GTFS limited; OSM-estimated transit'

    @staticmethod
    def _intent_instruction(language: str, origin: POI, dest: POI) -> str:
        o = origin.label.lower().replace('_', ' ')
        d = dest.label.lower().replace('_', ' ')
        if language == 'zh':
            return f'我在{o}门口，面朝主路。我想去{d}，请帮我规划完整路线。'
        if language == 'mixed':
            return f'我现在在{o}, facing the main road, and I need to reach {d}.'
        return f'I am outside {o}, facing the main road, and I need to reach {d}.'


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_region: dict[str, int] = {}
    direct_flights = 0
    one_stop_or_estimated = 0
    for ex in examples:
        by_split[ex['split']] = by_split.get(ex['split'], 0) + 1
        for region in ex.get('metadata', {}).get('macro_regions', []):
            if region:
                by_region[region] = by_region.get(region, 0) + 1
        path = ex.get('route_annotation', {}).get('flight_path') or []
        if len(path) == 2:
            direct_flights += 1
        elif path:
            one_stop_or_estimated += 1
    avg_instruction_len = sum(len(ex['instruction'].split()) for ex in examples) / max(1, len(examples))
    return {
        'count': len(examples),
        'by_split': by_split,
        'by_macro_region_mentions': by_region,
        'direct_flight_examples': direct_flights,
        'one_stop_or_estimated_flight_examples': one_stop_or_estimated,
        'avg_instruction_words': round(avg_instruction_len, 2),
        'max_city_pair_distance_km': round(_max_city_distance(examples), 1),
    }


def _max_city_distance(examples: list[dict[str, Any]]) -> float:
    max_dist = 0.0
    for ex in examples:
        gold = ex.get('gold_intent', {})
        origin = gold.get('origin') or {}
        dest = gold.get('destination') or {}
        if origin.get('lat') is None or dest.get('lat') is None:
            continue
        max_dist = max(max_dist, haversine_km(origin['lat'], origin['lon'], dest['lat'], dest['lon']))
    return max_dist

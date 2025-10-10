// frontend/src/components/RouteMap.tsx
import React from 'react';
import { MapContainer, TileLayer, Polyline } from 'react-leaflet';

interface Props {
  routeSegments: any[];
}

const RouteMap: React.FC<Props> = ({ routeSegments }) => {
  return (
    <MapContainer center={[28.61, 77.20]} zoom={13}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {routeSegments.map((seg, i) => (
        <Polyline key={i} pathOptions={{ color: seg.color }} positions={[[seg.start[0], seg.start[1]], [seg.end[0], seg.end[1]]]} />
      ))}
    </MapContainer>
  );
};

export default RouteMap;
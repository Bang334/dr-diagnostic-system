import assert from 'node:assert/strict';
import test from 'node:test';

import { resolvePatientFromQr } from './patientQrLookup.js';

test('resolves a scanned patient code to the full patient object', async () => {
  const expectedPatient = {
    id: 7,
    patient_code: 'BN1234',
    full_name: 'Nguyễn Văn A',
  };
  const searches = [];

  const patient = await resolvePatientFromQr('  bn1234\n', async (searchQuery) => {
    searches.push(searchQuery);
    return [
      { id: 8, patient_code: 'BN12345', full_name: 'Nguyễn Văn B' },
      expectedPatient,
    ];
  });

  assert.deepEqual(searches, ['bn1234']);
  assert.equal(patient, expectedPatient);
});

test('does not accept a fuzzy search result with a different patient code', async () => {
  const patient = await resolvePatientFromQr('BN1234', async () => [
    { id: 8, patient_code: 'BN12345', full_name: 'Nguyễn Văn B' },
  ]);

  assert.equal(patient, null);
});
